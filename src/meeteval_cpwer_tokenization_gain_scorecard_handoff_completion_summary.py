from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


COMPLETION_COLUMNS = [
    "scope",
    "handoff_status",
    "adapted_and_aligned_count",
    "case_count",
    "queue_status",
    "observation",
]


def load_handoff() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "meeteval_cpwer_tokenization_gain_scorecard_handoff.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_completion_row(handoff: dict[str, str]) -> dict[str, str]:
    handoff_status = str(handoff.get("handoff_status", "tokenization_gain_handoff_pending"))
    adapted_and_aligned_count = str(handoff.get("adapted_and_aligned_count", "0"))
    case_count = str(handoff.get("case_count", "0"))
    queue_status = (
        "queue_complete" if handoff_status == "tokenization_gain_handoff_ready" else "queue_in_progress"
    )
    return {
        "scope": "meeteval_cpwer_tokenization_gain_scorecard_handoff",
        "handoff_status": handoff_status,
        "adapted_and_aligned_count": adapted_and_aligned_count,
        "case_count": case_count,
        "queue_status": queue_status,
        "observation": (
            "Experimental/frontier tokenization gain handoff completion rollup; "
            "tokenization adaptation handoff remains the next coordination step."
        ),
    }


def build_completion_lines(row: dict[str, str]) -> list[str]:
    return [
        "# MeetEval cpWER Tokenization Gain Scorecard Handoff Completion Summary",
        "",
        "This generated note summarizes tokenization gain handoff completion. "
        "It does not claim full MeetEval benchmark completion.",
        "",
        "| scope | handoff_status | adapted_and_aligned_count | case_count | queue_status | observation |",
        "| --- | --- | ---: | ---: | --- | --- |",
        (
            f"| {row['scope']} | {row['handoff_status']} | {row['adapted_and_aligned_count']} | "
            f"{row['case_count']} | {row['queue_status']} | {row['observation']} |"
        ),
    ]


def write_outputs(completion_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_cpwer_tokenization_gain_scorecard_handoff_completion_summary.csv"
    json_path = tables_dir / "meeteval_cpwer_tokenization_gain_scorecard_handoff_completion_summary.json"
    md_path = figures_dir / "meeteval_cpwer_tokenization_gain_scorecard_handoff_completion_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COMPLETION_COLUMNS)
        writer.writeheader()
        writer.writerow(completion_row)
    json_path.write_text(json.dumps(completion_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_completion_lines(completion_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    handoff = load_handoff()
    if not handoff:
        print("Tokenization gain scorecard handoff not found; completion summary not written.")
        return
    completion_row = build_completion_row(handoff)
    csv_path, json_path, md_path = write_outputs(completion_row)
    print(
        "Wrote MeetEval cpWER tokenization gain scorecard handoff completion summary CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER tokenization gain scorecard handoff completion summary JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote MeetEval cpWER tokenization gain scorecard handoff completion summary note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Queue status: {completion_row['queue_status']}")


if __name__ == "__main__":
    main()
