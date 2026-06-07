from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


COMPLETION_COLUMNS = [
    "scope",
    "aligned_count",
    "total_count",
    "tokenization_root_cause_count",
    "queue_status",
    "observation",
]


def load_reconciliation_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_official_execution_reconciliation_audit.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def load_tokenization_diagnostic_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_official_execution_tokenization_diagnostic.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_completion_row(
    reconciliation_rows: list[dict[str, str]],
    diagnostic_rows: list[dict[str, str]],
) -> dict[str, str]:
    total_count = len(reconciliation_rows)
    aligned_count = sum(1 for row in reconciliation_rows if row.get("reconciliation_status") == "aligned")
    root_cause_count = sum(
        1 for row in diagnostic_rows if row.get("root_cause") == "no_whitespace_word_tokenization"
    )
    queue_status = "queue_complete" if total_count > 0 and aligned_count == total_count else "queue_in_progress"
    return {
        "scope": "meeteval_cpwer_tokenization_adaptation",
        "aligned_count": str(aligned_count),
        "total_count": str(total_count),
        "tokenization_root_cause_count": str(root_cause_count),
        "queue_status": queue_status,
        "observation": (
            "Experimental/frontier tokenization adaptation completion rollup; "
            "character-spaced official cpWER is bridge-lite comparable when root cause is addressed."
        ),
    }


def build_completion_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# MeetEval cpWER Tokenization Adaptation Completion Summary",
        "",
        "This generated note summarizes the tokenization adaptation stack completion. "
        "It does not claim full MeetEval benchmark completion.",
        "",
        "| scope | aligned_count | total_count | tokenization_root_cause_count | queue_status | observation |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        (
            f"| {row['scope']} | {row['aligned_count']} | {row['total_count']} | "
            f"{row['tokenization_root_cause_count']} | {row['queue_status']} | {row['observation']} |"
        ),
    ]
    return lines


def write_outputs(completion_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_tokenization_adaptation_completion_summary.csv"
    json_path = tables_dir / "meeteval_cpwer_tokenization_adaptation_completion_summary.json"
    md_path = figures_dir / "meeteval_cpwer_tokenization_adaptation_completion_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COMPLETION_COLUMNS)
        writer.writeheader()
        writer.writerow(completion_row)
    json_path.write_text(json.dumps(completion_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_completion_lines(completion_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    reconciliation_rows = load_reconciliation_rows()
    if not reconciliation_rows:
        print("Reconciliation audit not found; completion summary not written.")
        return
    completion_row = build_completion_row(reconciliation_rows, load_tokenization_diagnostic_rows())
    csv_path, json_path, md_path = write_outputs(completion_row)
    print(
        "Wrote MeetEval cpWER tokenization adaptation completion summary CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER tokenization adaptation completion summary JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER tokenization adaptation completion summary note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Queue status: {completion_row['queue_status']}")


if __name__ == "__main__":
    main()
