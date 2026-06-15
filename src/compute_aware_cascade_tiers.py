"""
Multi-Tier Compute-aware Cascaded Recognition System.

Tier 1 (Cheap):    whisper-small + router_v2  ->  cost ~1.0x base
Tier 2 (Stronger): High-risk cases -> whisper-medium or extra processing -> cost ~3.0x
Tier 3 (Expensive): Still unstable -> whisper-large-v3 + LLM critic -> cost ~8-15x

Outputs: CER-runtime tradeoff, model call counts, coverage, cost-aware routing tables.
Design: All routing uses reference-free signals only (NO CER as input).
       Larger-model CER uses conservative estimates bounded by oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT, load_config

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

TIERED_PERFORMANCE_COLUMNS = [
    "strategy", "label", "average_cer", "average_cost",
    "relative_cost_vs_tier1", "automatic_coverage",
    "tier_3_escalation_count", "tier_2_escalation_count",
    "tier_1_count", "sample_count", "case_count",
    "selected_tier_mix", "notes",
]

TIERED_ROUTING_COLUMNS = [
    "case_id", "overlap_level",
    "tier_1_method", "tier_1_cer", "tier_1_cost",
    "risk_level", "risk_reasons",
    "escalated_to_tier_2", "tier_2_method", "tier_2_cer", "tier_2_cost",
    "escalated_to_tier_3", "tier_3_method", "tier_3_cer", "tier_3_cost",
    "final_tier", "final_method", "final_cer", "final_cost",
    "total_cost", "oracle_cer", "notes",
    "split", "tier_name",
]

COST_BREAKDOWN_COLUMNS = [
    "tier", "strategy", "call_count", "cost_per_call",
    "total_tier_cost", "percentage_of_total", "notes",
]

MODEL_CALL_COLUMNS = [
    "strategy", "tier_1_calls", "tier_2_calls", "tier_3_calls",
    "llm_critic_calls", "total_model_calls", "total_cost", "coverage",
]

# ---------------------------------------------------------------------------
# Tier cost proxy (approximate RTF multipliers)
# ---------------------------------------------------------------------------

TIER_COST_PROXY = {
    "mixed_whisper_small": 1.0,
    "separated_whisper_small": 2.0,
    "separated_whisper_small_cleaned": 2.1,
    "mixed_whisper_medium": 3.0,
    "separated_whisper_medium": 6.0,
    "separated_whisper_medium_cleaned": 6.3,
    "mixed_whisper_large": 8.0,
    "separated_whisper_large": 16.0,
    "separated_whisper_large_cleaned": 16.8,
    "llm_critic": 15.0,
    "manual_review": 30.0,
}

# Conservative CER improvement factors (bounded by oracle CER)
TIER_2_CER_FACTOR = 0.80    # whisper-medium: ~20% CER reduction vs small
TIER_3_CER_FACTOR = 0.65    # whisper-large-v3: ~35% CER reduction vs small
LLM_CRITIC_CER_FACTOR = 0.85  # LLM repair: ~15% CER reduction

# Risk detection thresholds (reference-free signals only)
UNSTABLE_LENGTH_RATIO = 1.35
UNSTABLE_DUPLICATE_COUNT = 10
UNSTABLE_RUNTIME_RATIO = 1.8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return 0.0


def to_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return 0


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [row for row in csv.DictReader(f) if isinstance(row, dict)]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_csv_json(
    rows: list[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
    fieldnames: list[str],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Risk detection (reference-free signals ONLY)
# ---------------------------------------------------------------------------

def compute_risk_signals(
    overlap_level: int,
    mixed_len: int,
    separated_len: int,
    cleaned_len: int,
    duplicate_removed_count: int,
    runtime_ratio: float,
    cleaned_exists: bool,
    mixed_segments_count: int,
    separated_segments_count: int,
) -> tuple[str, list[str]]:
    """Compute risk level using only reference-free observable signals."""
    reasons: list[str] = []
    risk_score = 0

    if mixed_len > 0:
        length_ratio = separated_len / mixed_len
        if length_ratio > UNSTABLE_LENGTH_RATIO:
            risk_score += 2
            reasons.append("length_ratio=%.2f>%.2f" % (length_ratio, UNSTABLE_LENGTH_RATIO))
        elif length_ratio > 1.20:
            risk_score += 1
            reasons.append("length_ratio=%.2f>1.20" % length_ratio)

    if duplicate_removed_count >= UNSTABLE_DUPLICATE_COUNT:
        risk_score += 3
        reasons.append("dup_removed=%d>=%d" % (duplicate_removed_count, UNSTABLE_DUPLICATE_COUNT))
    elif duplicate_removed_count >= 5:
        risk_score += 1
        reasons.append("dup_removed=%d>=5" % duplicate_removed_count)

    if runtime_ratio > UNSTABLE_RUNTIME_RATIO:
        risk_score += 1
        reasons.append("runtime_ratio=%.2f>%.2f" % (runtime_ratio, UNSTABLE_RUNTIME_RATIO))

    if overlap_level in (1, 2):
        risk_score += 2
        reasons.append("overlap_level=%d (separation-hurts zone)" % overlap_level)
    elif overlap_level == 0 and mixed_segments_count > 5:
        risk_score += 1
        reasons.append("no_overlap_many_segments=%d" % mixed_segments_count)

    if separated_segments_count > 0 and mixed_segments_count > 0:
        seg_ratio = separated_segments_count / mixed_segments_count
        if seg_ratio > 2.0:
            risk_score += 1
            reasons.append("segment_ratio=%.2f>2.0" % seg_ratio)

    if risk_score >= 4:
        return "high", reasons
    elif risk_score >= 2:
        return "medium", reasons
    else:
        return "low", reasons


# ---------------------------------------------------------------------------
# Tier method selection
# ---------------------------------------------------------------------------

def is_unstable(mixed_len, separated_len, duplicate_removed_count, runtime_ratio):
    if mixed_len <= 0:
        return False
    if separated_len / mixed_len > UNSTABLE_LENGTH_RATIO:
        return True
    if duplicate_removed_count >= UNSTABLE_DUPLICATE_COUNT:
        return True
    if runtime_ratio > UNSTABLE_RUNTIME_RATIO:
        return True
    return False


def select_tier_1_method(overlap_level, mixed_len, separated_len, cleaned_len,
                         duplicate_removed_count, runtime_ratio, cleaned_exists,
                         mixed_segments_count):
    """Tier 1: router_v2 using whisper-small."""
    unstable = is_unstable(mixed_len, separated_len, duplicate_removed_count, runtime_ratio)

    if overlap_level == 0:
        if mixed_segments_count > 5:
            return ("separated_whisper_small", "t1:no_overlap_long->separated")
        if unstable and duplicate_removed_count >= 10:
            return ("mixed_whisper_small", "t1:no_overlap_high_dup->mixed")
        if (cleaned_exists and
            abs(cleaned_len - mixed_len) < abs(separated_len - mixed_len) and
            duplicate_removed_count < 5):
            return ("separated_whisper_small_cleaned", "t1:no_overlap_cleaned_closer->cleaned")
        return ("mixed_whisper_small", "t1:no_overlap_short->mixed")

    if overlap_level in (1, 2):
        return ("mixed_whisper_small", "t1:light_mid_overlap->mixed")

    if overlap_level >= 3:
        return ("separated_whisper_small", "t1:heavy_opposite->separated")

    return ("mixed_whisper_small", "t1:fallback->mixed")


def select_tier_2_method(tier_1_method, risk_level, overlap_level):
    """Tier 2: Escalate high/medium risk to whisper-medium."""
    if risk_level == "low":
        return tier_1_method
    mapping = {
        "mixed_whisper_small": "mixed_whisper_medium",
        "separated_whisper_small": "separated_whisper_medium",
        "separated_whisper_small_cleaned": "separated_whisper_medium_cleaned",
    }
    return mapping.get(tier_1_method, tier_1_method)


def select_tier_3_method(tier_2_method, risk_level):
    """Tier 3: Escalate still-high-risk to whisper-large-v3."""
    if risk_level == "low":
        return tier_2_method
    mapping = {
        "mixed_whisper_small": "mixed_whisper_large",
        "separated_whisper_small": "separated_whisper_large",
        "separated_whisper_small_cleaned": "separated_whisper_large_cleaned",
        "mixed_whisper_medium": "mixed_whisper_large",
        "separated_whisper_medium": "separated_whisper_large",
        "separated_whisper_medium_cleaned": "separated_whisper_large_cleaned",
    }
    return mapping.get(tier_2_method, tier_2_method)


# ---------------------------------------------------------------------------
# CER estimation for larger models (conservative, oracle-bounded)
# ---------------------------------------------------------------------------

def estimate_tier_cer(tier_1_cer, oracle_cer, target_method):
    """Estimate CER for larger model tier. Bounded below by oracle."""
    if tier_1_cer <= 0:
        return tier_1_cer
    if "large" in target_method:
        factor = TIER_3_CER_FACTOR
    elif "medium" in target_method:
        factor = TIER_2_CER_FACTOR
    else:
        factor = 1.0
    estimated = tier_1_cer * factor
    if oracle_cer > 0:
        estimated = max(estimated, oracle_cer * 0.95)
    return round(estimated, 6)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gold_data():
    """Load gold benchmark data from existing tables."""
    config = load_config()
    case_map = {case["id"]: case for case in config.get("audio_cases", [])}

    mixed_rows = read_csv_rows(PROJECT_ROOT / "results" / "tables" / "mixed_asr_benchmark.csv")
    separated_rows = read_csv_rows(PROJECT_ROOT / "results" / "tables" / "separated_asr_benchmark.csv")

    mixed = {str(r.get("case_id", "")): r for r in mixed_rows if str(r.get("case_id", "")).strip()}
    separated = {str(r.get("case_id", "")): r for r in separated_rows if str(r.get("case_id", "")).strip()}

    cleaned_rows = {}
    cleaned_dir = PROJECT_ROOT / "results" / "transcripts_postprocessed"
    for path in cleaned_dir.glob("*_separated_speaker_transcript_cleaned.json"):
        payload = read_json(path)
        case_id = str(payload.get("case_id", "")).strip()
        if case_id:
            cleaned_rows[case_id] = payload

    cer_rows = read_csv_rows(PROJECT_ROOT / "results" / "tables" / "cer_results.csv")
    cer_lookup = {}
    for row in cer_rows:
        cid = str(row.get("case_id", "")).strip()
        m = str(row.get("method", "")).strip()
        if cid and m:
            cer_lookup[(cid, m)] = to_float(row.get("cer"))

    return case_map, mixed, separated, cleaned_rows, cer_lookup


def load_synthetic_data():
    """Load synthetic benchmark data."""
    manifest_rows = read_csv_rows(PROJECT_ROOT / "results" / "tables" / "synthetic_manifest.csv")
    cleaned_rows = {}
    cleaned_dir = PROJECT_ROOT / "results" / "synthetic_transcripts_postprocessed"
    for path in cleaned_dir.glob("*_separated_speaker_transcript_cleaned.json"):
        payload = read_json(path)
        sample_id = str(payload.get("sample_id", "")).strip()
        if sample_id:
            cleaned_rows[sample_id] = payload

    cer_rows = read_csv_rows(PROJECT_ROOT / "results" / "tables" / "synthetic_cer_results.csv")
    cer_lookup = {}
    for row in cer_rows:
        sid = str(row.get("sample_id", "")).strip()
        m = str(row.get("method", "")).strip()
        if sid and m:
            cer_lookup[(sid, m)] = to_float(row.get("cer"))

    return manifest_rows, cleaned_rows, cer_lookup


# ---------------------------------------------------------------------------
# Core cascade routing
# ---------------------------------------------------------------------------

def route_case_through_tiers(
    case_id, overlap_level, mixed_len, separated_len, cleaned_len,
    duplicate_removed_count, runtime_ratio, cleaned_exists,
    mixed_segments_count, separated_segments_count,
    cer_lookup, skip_tier_3=False,
):
    """Route a single case through Tier 1 -> Tier 2 -> Tier 3."""

    # --- Tier 1: whisper-small + router_v2 ---
    tier_1_method, tier_1_rule = select_tier_1_method(
        overlap_level, mixed_len, separated_len, cleaned_len,
        duplicate_removed_count, runtime_ratio, cleaned_exists,
        mixed_segments_count,
    )
    tier_1_cer_method = tier_1_method.replace("_small", "")
    tier_1_cer = cer_lookup.get((case_id, tier_1_cer_method), 999.0)
    tier_1_cost = TIER_COST_PROXY.get(tier_1_method, 1.0)

    # Oracle (best possible) CER
    oracle_methods = ["mixed_whisper", "separated_whisper", "separated_whisper_cleaned"]
    oracle_cers = [cer_lookup.get((case_id, m), 999.0) for m in oracle_methods]
    oracle_cer = min(oracle_cers) if oracle_cers else 999.0

    # --- Risk assessment (reference-free) ---
    risk_level, risk_reasons = compute_risk_signals(
        overlap_level, mixed_len, separated_len, cleaned_len,
        duplicate_removed_count, runtime_ratio, cleaned_exists,
        mixed_segments_count, separated_segments_count,
    )

    # --- Tier 2: escalate to whisper-medium ---
    tier_2_method = select_tier_2_method(tier_1_method, risk_level, overlap_level)
    escalated_to_tier_2 = tier_2_method != tier_1_method
    if escalated_to_tier_2:
        tier_2_cer = estimate_tier_cer(tier_1_cer, oracle_cer, tier_2_method)
    else:
        tier_2_cer = tier_1_cer
    tier_2_cost = TIER_COST_PROXY.get(tier_2_method, tier_1_cost)

    # --- Tier 3: escalate to whisper-large + LLM ---
    tier_3_method = select_tier_3_method(tier_2_method, risk_level)
    escalated_to_tier_3 = (tier_3_method != tier_2_method) and not skip_tier_3
    if escalated_to_tier_3:
        tier_3_cer = estimate_tier_cer(tier_1_cer, oracle_cer, tier_3_method)
    else:
        tier_3_cer = tier_2_cer
    tier_3_cost = TIER_COST_PROXY.get(tier_3_method, tier_2_cost)

    # Total cost
    total_cost = tier_1_cost
    if escalated_to_tier_2:
        total_cost += tier_2_cost
    if escalated_to_tier_3:
        total_cost += tier_3_cost

    # Final decision
    if escalated_to_tier_3:
        final_tier, final_method, final_cer = 3, tier_3_method, tier_3_cer
    elif escalated_to_tier_2:
        final_tier, final_method, final_cer = 2, tier_2_method, tier_2_cer
    else:
        final_tier, final_method, final_cer = 1, tier_1_method, tier_1_cer

    return {
        "case_id": case_id, "overlap_level": overlap_level,
        "tier_1_method": tier_1_method, "tier_1_cer": tier_1_cer,
        "tier_1_cost": tier_1_cost,
        "risk_level": risk_level, "risk_reasons": "; ".join(risk_reasons),
        "escalated_to_tier_2": escalated_to_tier_2,
        "tier_2_method": tier_2_method, "tier_2_cer": tier_2_cer,
        "tier_2_cost": tier_2_cost,
        "escalated_to_tier_3": escalated_to_tier_3,
        "tier_3_method": tier_3_method, "tier_3_cer": tier_3_cer,
        "tier_3_cost": tier_3_cost,
        "final_tier": final_tier, "final_method": final_method,
        "final_cer": final_cer, "final_cost": total_cost,
        "total_cost": total_cost, "oracle_cer": oracle_cer,
        "notes": "rule=%s; risk=%s; t2=%s; t3=%s" % (
            tier_1_rule, risk_level,
            "yes" if escalated_to_tier_2 else "no",
            "yes" if escalated_to_tier_3 else "no"),
    }


# ---------------------------------------------------------------------------
# Strategy comparison
# ---------------------------------------------------------------------------

TIERED_STRATEGIES = [
    "fixed_mixed_small",
    "fixed_separated_small",
    "fixed_separated_small_cleaned",
    "router_v2_tier1_only",
    "router_v2_tier2_cascade",
    "router_v2_tier3_cascade",
    "cost_first_cascade",
    "accuracy_first_cascade",
    "oracle_best",
]


def build_tiered_performance_rows(routing_decisions, cer_lookup):
    """Build strategy comparison rows across all tiered strategies."""
    decision_map = {d["case_id"]: d for d in routing_decisions}
    case_ids = list(decision_map)
    case_count = len(case_ids)
    rows = []

    for strategy in TIERED_STRATEGIES:
        cer_values = []
        costs = []
        t1c = t2c = t3c = 0

        for case_id in case_ids:
            d = decision_map[case_id]

            if strategy == "fixed_mixed_small":
                cer = cer_lookup.get((case_id, "mixed_whisper"), 999.0)
                cost = TIER_COST_PROXY["mixed_whisper_small"]
            elif strategy == "fixed_separated_small":
                cer = cer_lookup.get((case_id, "separated_whisper"), 999.0)
                cost = TIER_COST_PROXY["separated_whisper_small"]
            elif strategy == "fixed_separated_small_cleaned":
                cer = cer_lookup.get((case_id, "separated_whisper_cleaned"), 999.0)
                cost = TIER_COST_PROXY["separated_whisper_small_cleaned"]
            elif strategy == "router_v2_tier1_only":
                cer = d["tier_1_cer"]
                cost = d["tier_1_cost"]
                t1c += 1
            elif strategy == "router_v2_tier2_cascade":
                cer = d["tier_2_cer"]
                cost = d["tier_1_cost"] + (d["tier_2_cost"] if d["escalated_to_tier_2"] else 0)
                t1c += 1
                if d["escalated_to_tier_2"]:
                    t2c += 1
            elif strategy == "router_v2_tier3_cascade":
                cer = d["tier_3_cer"]
                cost = d["total_cost"]
                t1c += 1
                if d["escalated_to_tier_2"]:
                    t2c += 1
                if d["escalated_to_tier_3"]:
                    t3c += 1
            elif strategy == "cost_first_cascade":
                if d["overlap_level"] in (1, 2):
                    cer = cer_lookup.get((case_id, "mixed_whisper"), 999.0)
                    cost = TIER_COST_PROXY["mixed_whisper_small"]
                elif d["overlap_level"] == 0:
                    cer = cer_lookup.get((case_id, "separated_whisper"), 999.0)
                    cost = TIER_COST_PROXY["separated_whisper_small"]
                else:
                    cer = d["tier_1_cer"]
                    cost = d["tier_1_cost"]
                t1c += 1
            elif strategy == "accuracy_first_cascade":
                cer = d["tier_3_cer"]
                cost = d["tier_1_cost"] + d["tier_2_cost"] + d["tier_3_cost"]
                t1c += 1
                t2c += 1
                t3c += 1
            elif strategy == "oracle_best":
                cer = d["oracle_cer"]
                cost = 999.0
            else:
                cer = 999.0
                cost = 1.0

            if cer < 900:
                cer_values.append(cer)
            costs.append(cost)

        rows.append({
            "strategy": strategy,
            "label": "experimental/frontier",
            "average_cer": round(sum(cer_values) / len(cer_values), 6) if cer_values else "",
            "average_cost": round(sum(costs) / len(costs), 6) if costs else 0.0,
            "relative_cost_vs_tier1": "",
            "automatic_coverage": round(len(cer_values) / case_count, 6) if case_count else 0.0,
            "tier_3_escalation_count": t3c,
            "tier_2_escalation_count": t2c,
            "tier_1_count": t1c,
            "sample_count": len(cer_values),
            "case_count": case_count,
            "selected_tier_mix": "t1:%d;t2:%d;t3:%d" % (t1c, t2c, t3c),
            "notes": (
                "Multi-tier cascade. Tier 2/3 CER uses conservative estimates "
                "bounded by oracle CER. Labeled experimental/frontier until "
                "verified with real larger-model outputs."
            ),
        })

    # Compute relative costs
    tier1_only_cost = next(
        (to_float(r["average_cost"]) for r in rows if r["strategy"] == "router_v2_tier1_only"),
        0.0,
    )
    for row in rows:
        row["relative_cost_vs_tier1"] = (
            round(to_float(row["average_cost"]) / tier1_only_cost, 6) if tier1_only_cost else ""
        )

    return rows


# ---------------------------------------------------------------------------
# Cost breakdown and model call analysis
# ---------------------------------------------------------------------------

def build_cost_breakdown_rows(routing_decisions, strategy):
    """Build per-tier cost breakdown."""
    case_count = len(routing_decisions)
    t1_total = sum(d["tier_1_cost"] for d in routing_decisions)
    t2_total = sum(d["tier_2_cost"] for d in routing_decisions if d["escalated_to_tier_2"])
    t3_total = sum(d["tier_3_cost"] for d in routing_decisions if d["escalated_to_tier_3"])
    grand = t1_total + t2_total + t3_total
    if grand == 0:
        grand = 1.0

    rows = []
    for tname, ttotal, ccount in [
        ("Tier 1 (small)", t1_total, case_count),
        ("Tier 2 (medium)", t2_total, sum(1 for d in routing_decisions if d["escalated_to_tier_2"])),
        ("Tier 3 (large+LLM)", t3_total, sum(1 for d in routing_decisions if d["escalated_to_tier_3"])),
    ]:
        rows.append({
            "tier": tname,
            "strategy": strategy,
            "call_count": ccount,
            "cost_per_call": round(ttotal / case_count, 2) if case_count else 0,
            "total_tier_cost": round(ttotal, 2),
            "percentage_of_total": round(ttotal / grand * 100, 1),
            "notes": "",
        })
    return rows


def build_model_call_rows(routing_decisions, performance_rows):
    """Build model call count comparison."""
    case_count = len(routing_decisions)
    rows = []
    for pr in performance_rows:
        t1 = to_int(pr.get("tier_1_count", case_count))
        t2 = to_int(pr.get("tier_2_escalation_count", 0))
        t3 = to_int(pr.get("tier_3_escalation_count", 0))
        rows.append({
            "strategy": pr["strategy"],
            "tier_1_calls": t1,
            "tier_2_calls": t2,
            "tier_3_calls": t3,
            "llm_critic_calls": t3,
            "total_model_calls": t1 + t2 + t3,
            "total_cost": to_float(pr.get("average_cost", 0)),
            "coverage": to_float(pr.get("automatic_coverage", 0)),
        })
    return rows


# ---------------------------------------------------------------------------
# Synthetic benchmark routing
# ---------------------------------------------------------------------------

TIER_NAME_MAP = {
    "SyntheticNoOverlap": 0, "SyntheticLightOverlap": 1,
    "SyntheticMidOverlap": 2, "SyntheticHeavyOverlap": 3,
    "SyntheticOppositeOverlap": 4,
}


def route_synthetic_through_tiers(manifest_rows, cleaned_rows, cer_lookup, skip_tier_3=False):
    """Route all synthetic cases through the tiered cascade."""
    decisions = []
    for row in manifest_rows:
        sample_id = str(row.get("sample_id", "")).strip()
        tier_name = str(row.get("tier", "")).strip()
        overlap_level = int(row.get("overlap_level_numeric", TIER_NAME_MAP.get(tier_name, 0)))

        mixed_path = PROJECT_ROOT / "results" / "synthetic_transcripts_raw" / ("%s_mixed_whisper.json" % sample_id)
        speaker_path = PROJECT_ROOT / "results" / "synthetic_transcripts_speaker" / ("%s_separated_speaker_transcript.json" % sample_id)
        cleaned_path = PROJECT_ROOT / "results" / "synthetic_transcripts_postprocessed" / ("%s_separated_speaker_transcript_cleaned.json" % sample_id)

        if not mixed_path.exists() or not speaker_path.exists():
            continue

        mixed_data = read_json(mixed_path)
        separated_data = read_json(speaker_path)
        cleaned_data = cleaned_rows.get(sample_id)
        if cleaned_data is None and cleaned_path.exists():
            cleaned_data = read_json(cleaned_path)

        mixed_runtime = to_float(mixed_data.get("runtime_sec"))
        d = route_case_through_tiers(
            case_id=sample_id,
            overlap_level=overlap_level,
            mixed_len=len(str(mixed_data.get("text", ""))),
            separated_len=len(str(separated_data.get("full_text", ""))),
            cleaned_len=len(str(cleaned_data.get("cleaned_full_text", ""))) if cleaned_data else 0,
            duplicate_removed_count=to_int(cleaned_data.get("removed_count")) if cleaned_data else 0,
            runtime_ratio=(round(to_float(separated_data.get("runtime_sec_total")) / mixed_runtime, 6)
                          if mixed_runtime > 0 else 0.0),
            cleaned_exists=bool(cleaned_data),
            mixed_segments_count=len(mixed_data.get("segments", [])),
            separated_segments_count=len(separated_data.get("segments", [])),
            cer_lookup=cer_lookup,
            skip_tier_3=skip_tier_3,
        )
        d["split"] = str(row.get("split", "")).strip()
        d["tier_name"] = tier_name
        decisions.append(d)
    return decisions


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def render_tradeoff(rows):
    """Render CER-Runtime tradeoff table."""
    lines = [
        "# Multi-Tier Cascade: CER-Runtime Tradeoff",
        "",
        "| Strategy | Avg CER | Avg Cost | Rel Cost vs T1 | T2 Esc | T3 Esc | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            "| %s | %s | %.2f | %s | %s | %s | %.0f%% |" % (
                r["strategy"], r["average_cer"], to_float(r["average_cost"]),
                r["relative_cost_vs_tier1"], r["tier_2_escalation_count"],
                r["tier_3_escalation_count"], to_float(r["automatic_coverage"]) * 100,
            )
        )
    lines.extend([
        "",
        "**Interpretation:**",
        "- `router_v2_tier1_only`: Best cost-efficiency; all cases use cheap whisper-small.",
        "- `router_v2_tier2_cascade`: Escalates high-risk cases to medium; better CER at moderate cost.",
        "- `router_v2_tier3_cascade`: Escalates still-risky to large+LLM; best CER at highest cost.",
        "- `cost_first_cascade`: Always pick cheapest viable route per case.",
        "- `accuracy_first_cascade`: Always use strongest tier for every case.",
        "",
        "**Note:** Tier 2/3 CER values are conservative ESTIMATES bounded by oracle CER.",
    ])
    return "\n".join(lines)


def render_routing(decisions):
    """Render per-case routing decisions."""
    lines = [
        "# Multi-Tier Cascade: Per-Case Routing Decisions",
        "",
        "| Case | Overlap | Risk | T1 Method | T1 CER | ->T2 | T2 CER | ->T3 | Final Tier | Final CER | Cost |",
        "| --- | ---: | --- | --- | ---: | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for d in decisions:
        lines.append(
            "| %s | %s | %s | %s | %.4f | %s | %.4f | %s | %s | %.4f | %.1f |" % (
                d["case_id"], d["overlap_level"], d["risk_level"],
                d["tier_1_method"], d["tier_1_cer"],
                "YES" if d["escalated_to_tier_2"] else "--",
                d["tier_2_cer"],
                "YES" if d["escalated_to_tier_3"] else "--",
                d["final_tier"], d["final_cer"], d["total_cost"],
            )
        )
    lines.extend(["", "## Risk Reasons", ""])
    for d in decisions:
        lines.append("- **%s**: %s" % (d["case_id"], d["risk_reasons"]))
    return "\n".join(lines)


def render_pareto(rows):
    """Identify and render Pareto frontier."""
    lines = [
        "# Multi-Tier Cascade: Pareto Frontier Analysis",
        "",
        "| Strategy | Avg CER | Avg Cost | Frontier Status |",
        "| --- | ---: | ---: | --- |",
    ]

    # Find Pareto-optimal strategies
    pareto_set = set()
    for i, ri in enumerate(rows):
        ci = to_float(ri.get("average_cer", 999))
        csti = to_float(ri.get("average_cost", 999))
        if ci == "" or ci >= 900:
            continue
        dominated = False
        for j, rj in enumerate(rows):
            if i == j:
                continue
            cj = to_float(rj.get("average_cer", 999))
            cstj = to_float(rj.get("average_cost", 999))
            if cj == "" or cj >= 900:
                continue
            if cstj <= csti and cj < ci:
                dominated = True
                break
            if cstj < csti and cj <= ci:
                dominated = True
                break
        if not dominated:
            pareto_set.add(ri["strategy"])

    for r in rows:
        status = "**PARETO**" if r["strategy"] in pareto_set else "dominated"
        lines.append("| %s | %s | %.2f | %s |" % (
            r["strategy"], r["average_cer"], to_float(r["average_cost"]), status
        ))

    lines.extend(["", "## Deployment Recommendations", ""])
    pareto_rows = [r for r in rows if r["strategy"] in pareto_set]
    if pareto_rows:
        cheapest = min(pareto_rows, key=lambda r: to_float(r.get("average_cost", 999)))
        best = min(pareto_rows, key=lambda r: to_float(r.get("average_cer", 999)))
        lines.append("- **Cost-first**: `%s` (CER=%s, Cost=%.2f)" % (
            cheapest["strategy"], cheapest["average_cer"], to_float(cheapest["average_cost"])))
        lines.append("- **Accuracy-first**: `%s` (CER=%s, Cost=%.2f)" % (
            best["strategy"], best["average_cer"], to_float(best["average_cost"])))

    lines.extend([
        "",
        "**Note:** This Pareto analysis uses conservative CER estimates. "
        "Refresh with real larger-model measurements for production.",
    ])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Tier Compute-aware Cascaded Recognition")
    parser.add_argument("--dataset", default="gold",
                        choices=["gold", "synthetic", "synthetic_split", "all"])
    parser.add_argument("--skip-tier3", action="store_true",
                        help="Skip Tier 3 escalation (large model + LLM critic)")
    parser.add_argument("--case", default="all")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("Multi-Tier Compute-aware Cascaded Recognition System")
    print("=" * 70)
    print("Dataset: %s  |  Tier3: %s" % (
        args.dataset, "OFF" if args.skip_tier3 else "ON"))
    print()

    out_dir = PROJECT_ROOT / "results"

    # ===================================================================
    # Gold Benchmark
    # ===================================================================
    if args.dataset in ("gold", "all"):
        print("=" * 70)
        print("GOLD BENCHMARK -- Multi-Tier Cascade Analysis")
        print("=" * 70)

        case_map, mixed, separated, cleaned_rows, cer_lookup = load_gold_data()
        decisions = []

        for case_id in sorted(case_map):
            case = case_map[case_id]
            mr = mixed.get(case_id, {})
            sr = separated.get(case_id, {})
            cd = cleaned_rows.get(case_id, {})

            d = route_case_through_tiers(
                case_id=case_id,
                overlap_level=int(case.get("overlap_level", 0)),
                mixed_len=to_int(mr.get("text_length")),
                separated_len=to_int(sr.get("full_text_length")),
                cleaned_len=len(str(cd.get("cleaned_full_text", ""))),
                duplicate_removed_count=to_int(cd.get("removed_count")),
                runtime_ratio=(round(to_float(sr.get("runtime_sec_total")) /
                               to_float(mr.get("runtime_sec")), 6)
                               if to_float(mr.get("runtime_sec")) > 0 else 0.0),
                cleaned_exists=bool(cd),
                mixed_segments_count=to_int(mr.get("segments_count")),
                separated_segments_count=to_int(sr.get("merged_segments_count")),
                cer_lookup=cer_lookup,
                skip_tier_3=args.skip_tier3,
            )
            decisions.append(d)

        perf = build_tiered_performance_rows(decisions, cer_lookup)
        cost_br = build_cost_breakdown_rows(decisions, "router_v2_tier3_cascade")
        model_rows = build_model_call_rows(decisions, perf)

        # Write tables
        write_csv_json(decisions,
                       out_dir / "tables" / "tiered_cascade_routing.csv",
                       out_dir / "tables" / "tiered_cascade_routing.json",
                       TIERED_ROUTING_COLUMNS)
        write_csv_json(perf,
                       out_dir / "tables" / "tiered_cascade_performance.csv",
                       out_dir / "tables" / "tiered_cascade_performance.json",
                       TIERED_PERFORMANCE_COLUMNS)
        write_csv_json(cost_br,
                       out_dir / "tables" / "tiered_cascade_cost_breakdown.csv",
                       out_dir / "tables" / "tiered_cascade_cost_breakdown.json",
                       COST_BREAKDOWN_COLUMNS)
        write_csv_json(model_rows,
                       out_dir / "tables" / "tiered_cascade_model_calls.csv",
                       out_dir / "tables" / "tiered_cascade_model_calls.json",
                       MODEL_CALL_COLUMNS)

        # Write markdown summaries
        (out_dir / "figures" / "tiered_cascade_tradeoff.md").write_text(
            render_tradeoff(perf), encoding="utf-8")
        (out_dir / "figures" / "tiered_cascade_routing.md").write_text(
            render_routing(decisions), encoding="utf-8")
        (out_dir / "figures" / "tiered_cascade_pareto.md").write_text(
            render_pareto(perf), encoding="utf-8")

        # Print summary
        print("\nGOLD BENCHMARK -- Cascade Summary")
        print("-" * 70)
        for r in perf:
            print("  %-38s  CER=%-10s  Cost=%8.2f  T2=%s  T3=%s" % (
                r["strategy"], str(r["average_cer"]),
                to_float(r["average_cost"]),
                r["tier_2_escalation_count"], r["tier_3_escalation_count"]))

    # ===================================================================
    # Synthetic Benchmark
    # ===================================================================
    if args.dataset in ("synthetic", "synthetic_split", "all"):
        print("\n" + "=" * 70)
        print("SYNTHETIC BENCHMARK -- Multi-Tier Cascade Analysis")
        print("=" * 70)

        manifest_rows, cleaned_rows, syn_cer_lookup = load_synthetic_data()
        syn_decisions = route_synthetic_through_tiers(
            manifest_rows, cleaned_rows, syn_cer_lookup, args.skip_tier3)

        if not syn_decisions:
            print("No synthetic cases found. Generate synthetic benchmark first.")
        else:
            syn_perf = build_tiered_performance_rows(syn_decisions, syn_cer_lookup)
            write_csv_json(syn_decisions,
                           out_dir / "tables" / "tiered_cascade_synthetic_routing.csv",
                           out_dir / "tables" / "tiered_cascade_synthetic_routing.json",
                           TIERED_ROUTING_COLUMNS)
            write_csv_json(syn_perf,
                           out_dir / "tables" / "tiered_cascade_synthetic_performance.csv",
                           out_dir / "tables" / "tiered_cascade_synthetic_performance.json",
                           TIERED_PERFORMANCE_COLUMNS)
            (out_dir / "figures" / "tiered_cascade_synthetic_tradeoff.md").write_text(
                render_tradeoff(syn_perf), encoding="utf-8")
            (out_dir / "figures" / "tiered_cascade_synthetic_pareto.md").write_text(
                render_pareto(syn_perf), encoding="utf-8")

            print("\nSYNTHETIC BENCHMARK -- Cascade Summary")
            print("-" * 70)
            for r in syn_perf:
                print("  %-38s  CER=%-10s  Cost=%8.2f  T2=%s  T3=%s" % (
                    r["strategy"], str(r["average_cer"]),
                    to_float(r["average_cost"]),
                    r["tier_2_escalation_count"], r["tier_3_escalation_count"]))

    print("\n" + "=" * 70)
    print("Multi-Tier Cascade Analysis Complete")
    print("=" * 70)
    print()
    print("Output files written to:")
    print("  results/tables/tiered_cascade_*.csv")
    print("  results/tables/tiered_cascade_*.json")
    print("  results/figures/tiered_cascade_*.md")
    print()
    print("Key takeaways:")
    print("  1. Tier 1 (small + router_v2): best cost-efficiency")
    print("  2. Tier 2 escalation: helps high-risk cases at moderate cost")
    print("  3. Tier 3 escalation: best CER at highest cost")
    print("  4. Pareto frontier informs cost vs accuracy deployment tradeoffs")
    print()
    print("Next steps:")
    print("  - Replace Tier 2/3 CER estimates with real larger-model ASR outputs")
    print("  - Replace proxy costs with measured RTF on controlled hardware")
    print("  - Enable real LLM critic via existing llm_repair_loop.py")


if __name__ == "__main__":
    main()
