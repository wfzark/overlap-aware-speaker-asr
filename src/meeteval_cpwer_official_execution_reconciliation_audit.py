from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT
from .meeteval_cpwer_official_execution_alignment_audit import (
    classify_alignment,
    compute_alignment_delta,
    load_bridge_lite_by_case,
)


RECONCILIATION_COLUMNS = [
    "case_id",
    "character_level_cpwer",
    "cpwer_bridge_lite",
    "reconciliation_delta",
    "reconciliation_status",
    "audit_note",
]


def load_character_level_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_character_level_official_execution.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_reconciliation_rows(
    char_rows: list[dict[str, str]],
    bridge_lite_by_case: dict[str, str],
) -> list[dict[str, str]]:
    reconciliation_rows: list[dict[str, str]] = []
    for row in char_rows:
        case_id = str(row.get("case_id", ""))
        char_cpwer = str(row.get("official_cpwer", ""))
        bridge_lite = bridge_lite_by_case.get(case_id, "")
        delta = compute_alignment_delta(char_cpwer, bridge_lite)
        status = classify_alignment(delta)
        if not char_cpwer:
            audit_note = "Character-level official cpWER not yet available for reconciliation."
        elif status == "aligned":
            audit_note = "Character-spaced MeetEval cpWER aligns with bridge-lite within tolerance."
        elif status == "minor_drift":
            audit_note = "Minor residual drift after character tokenization; review speaker mapping or normalization."
        elif status == "moderate_drift":
            audit_note = "Moderate drift persists after character tokenization; inspect segment aggregation."
        else:
            audit_note = "Reconciliation audit pending character-level execution."
        reconciliation_rows.append(
            {
                "case_id": case_id,
                "character_level_cpwer": char_cpwer,
                "cpwer_bridge_lite": bridge_lite,
                "reconciliation_delta": delta,
                "reconciliation_status": status,
                "audit_note": audit_note,
            }
        )
    return reconciliation_rows


def build_reconciliation_lines(rows: list[dict[str, str]]) -> list[str]:
    aligned_count = sum(1 for row in rows if row.get("reconciliation_status") == "aligned")
    minor_count = sum(1 for row in rows if row.get("reconciliation_status") == "minor_drift")
    lines = [
        "# MeetEval cpWER Official Execution Reconciliation Audit",
        "",
        "This generated audit compares character-spaced MeetEval cpWER against bridge-lite after tokenization adaptation. "
        "Results remain experimental/frontier and do not constitute a full benchmark claim.",
        "",
        f"Summary: `{aligned_count}/{len(rows)}` aligned, `{minor_count}/{len(rows)}` minor drift (delta <= 0.05).",
        "",
        "| case_id | character_level_cpwer | cpwer_bridge_lite | reconciliation_delta | reconciliation_status | audit_note |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        char_display = row["character_level_cpwer"] if row["character_level_cpwer"] else "—"
        bridge_display = row["cpwer_bridge_lite"] if row["cpwer_bridge_lite"] else "—"
        delta_display = row["reconciliation_delta"] if row["reconciliation_delta"] else "—"
        lines.append(
            f"| {row['case_id']} | {char_display} | {bridge_display} | {delta_display} | "
            f"{row['reconciliation_status']} | {row['audit_note']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_official_execution_reconciliation_audit.csv"
    json_path = tables_dir / "meeteval_cpwer_official_execution_reconciliation_audit.json"
    md_path = figures_dir / "meeteval_cpwer_official_execution_reconciliation_audit.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RECONCILIATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_reconciliation_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    char_rows = load_character_level_rows()
    if not char_rows:
        print("Character-level official execution output not found; reconciliation audit not written.")
        return
    rows = build_reconciliation_rows(char_rows, load_bridge_lite_by_case())
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote MeetEval cpWER official execution reconciliation audit CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER official execution reconciliation audit JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER official execution reconciliation audit note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
