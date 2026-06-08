from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


ALIGNMENT_COLUMNS = [
    "case_id",
    "official_cpwer",
    "cpwer_bridge_lite",
    "alignment_delta",
    "alignment_status",
    "audit_note",
]


def load_official_execution_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_official_execution.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def load_bridge_lite_by_case() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_bridge.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {str(row.get("case_id", "")): str(row.get("cpwer_bridge_lite", "")) for row in reader}


def load_json_rows_by_case(path_rel: str, case_key: str = "case_id") -> dict[str, dict[str, str]]:
    path = PROJECT_ROOT / path_rel
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return {}
    rows_by_case: dict[str, dict[str, str]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get(case_key, "")).strip()
        if case_id:
            rows_by_case[case_id] = {str(k): str(v) for k, v in row.items()}
    return rows_by_case


def compute_alignment_delta(official: str, bridge_lite: str) -> str:
    if not official or not bridge_lite:
        return ""
    try:
        return str(round(float(official) - float(bridge_lite), 6))
    except ValueError:
        return ""


def classify_alignment(delta: str) -> str:
    if not delta:
        return "pending"
    try:
        abs_delta = abs(float(delta))
    except ValueError:
        return "pending"
    if abs_delta <= 0.01:
        return "aligned"
    if abs_delta <= 0.05:
        return "minor_drift"
    return "moderate_drift"


def build_alignment_rows(
    execution_rows: list[dict[str, str]],
    bridge_lite_by_case: dict[str, str],
) -> list[dict[str, str]]:
    tokenization_by_case = load_json_rows_by_case(
        "results/tables/meeteval_cpwer_official_execution_tokenization_diagnostic.json"
    )
    reconciliation_by_case = load_json_rows_by_case(
        "results/tables/meeteval_cpwer_official_execution_reconciliation_audit.json"
    )
    alignment_rows: list[dict[str, str]] = []
    for row in execution_rows:
        case_id = str(row.get("case_id", ""))
        official_cpwer = str(row.get("official_cpwer", ""))
        bridge_lite = bridge_lite_by_case.get(case_id, "")
        delta = compute_alignment_delta(official_cpwer, bridge_lite)
        alignment_status = classify_alignment(delta)
        tokenization_row = tokenization_by_case.get(case_id, {})
        reconciliation_row = reconciliation_by_case.get(case_id, {})
        tokenization_root_cause = tokenization_row.get("root_cause", "")
        reconciliation_status = reconciliation_row.get("reconciliation_status", "")
        if not official_cpwer:
            audit_note = "Official cpWER not yet available for alignment audit."
        elif alignment_status == "aligned":
            audit_note = "Official cpWER and bridge-lite scores are closely aligned."
        elif alignment_status == "minor_drift":
            audit_note = "Minor drift between official cpWER and bridge-lite; review segment aggregation."
        elif alignment_status == "moderate_drift":
            if tokenization_root_cause == "no_whitespace_word_tokenization":
                if reconciliation_status == "aligned":
                    audit_note = (
                        "Moderate drift is explained by Chinese word-level tokenization mismatch in raw MeetEval cpWER; "
                        "character-spaced reconciliation already realigns with bridge-lite."
                    )
                else:
                    audit_note = (
                        "Moderate drift is consistent with Chinese word-level tokenization mismatch; "
                        "verify character-spaced reconciliation before escalating mapping concerns."
                    )
            else:
                audit_note = "Moderate drift detected; segment or speaker mapping may need inspection."
        else:
            audit_note = "Alignment audit pending official cpWER execution."
        alignment_rows.append(
            {
                "case_id": case_id,
                "official_cpwer": official_cpwer,
                "cpwer_bridge_lite": bridge_lite,
                "alignment_delta": delta,
                "alignment_status": alignment_status,
                "audit_note": audit_note,
            }
        )
    return alignment_rows


def build_alignment_lines(rows: list[dict[str, str]]) -> list[str]:
    aligned_count = sum(1 for row in rows if row.get("alignment_status") == "aligned")
    lines = [
        "# MeetEval cpWER Official Execution Alignment Audit",
        "",
        "This generated audit compares official MeetEval cpWER scores against the bridge-lite baseline. "
        "Results remain experimental/frontier and do not constitute a full benchmark claim.",
        "",
        f"Summary: `{aligned_count}/{len(rows)}` cases report aligned scores (delta <= 0.01).",
        "",
        "| case_id | official_cpwer | cpwer_bridge_lite | alignment_delta | alignment_status | audit_note |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        official_display = row["official_cpwer"] if row["official_cpwer"] else "—"
        bridge_display = row["cpwer_bridge_lite"] if row["cpwer_bridge_lite"] else "—"
        delta_display = row["alignment_delta"] if row["alignment_delta"] else "—"
        lines.append(
            f"| {row['case_id']} | {official_display} | {bridge_display} | {delta_display} | "
            f"{row['alignment_status']} | {row['audit_note']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_official_execution_alignment_audit.csv"
    json_path = tables_dir / "meeteval_cpwer_official_execution_alignment_audit.json"
    md_path = figures_dir / "meeteval_cpwer_official_execution_alignment_audit.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ALIGNMENT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_alignment_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    execution_rows = load_official_execution_rows()
    if not execution_rows:
        print("Official execution output not found; alignment audit not written.")
        return
    rows = build_alignment_rows(execution_rows, load_bridge_lite_by_case())
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote MeetEval cpWER official execution alignment audit CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER official execution alignment audit JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER official execution alignment audit note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
