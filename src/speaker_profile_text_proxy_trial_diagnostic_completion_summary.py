from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


COMPLETION_COLUMNS = [
    "scope",
    "swapped_count",
    "case_count",
    "average_confidence_gap",
    "queue_status",
    "observation",
]


def load_diagnostic_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_text_proxy_trial_diagnostic_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_completion_row(summary: dict[str, str]) -> dict[str, str]:
    case_count = int(summary.get("case_count", "0") or 0)
    swapped_count = int(summary.get("swapped_count", "0") or 0)
    queue_status = "queue_complete" if case_count > 0 and swapped_count == case_count else "queue_in_progress"
    return {
        "scope": "speaker_profile_text_proxy_trial",
        "swapped_count": str(swapped_count),
        "case_count": str(case_count),
        "average_confidence_gap": str(summary.get("average_confidence_gap", "0")),
        "queue_status": queue_status,
        "observation": (
            "Experimental/frontier text-proxy diagnostic completion rollup; "
            "swapped bias confirms text profile is a warning sign, not deployment-ready attribution."
        ),
    }


def build_completion_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Speaker Profile Text-Proxy Trial Diagnostic Completion Summary",
        "",
        "This generated note summarizes the all-gold text-proxy diagnostic completion. "
        "It does not claim voiceprint success or improved speaker attribution.",
        "",
        "| scope | swapped_count | case_count | average_confidence_gap | queue_status | observation |",
        "| --- | ---: | ---: | ---: | --- | --- |",
        (
            f"| {row['scope']} | {row['swapped_count']} | {row['case_count']} | "
            f"{row['average_confidence_gap']} | {row['queue_status']} | {row['observation']} |"
        ),
    ]


def write_outputs(completion_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_text_proxy_trial_diagnostic_completion_summary.csv"
    json_path = tables_dir / "speaker_profile_text_proxy_trial_diagnostic_completion_summary.json"
    md_path = figures_dir / "speaker_profile_text_proxy_trial_diagnostic_completion_summary.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COMPLETION_COLUMNS)
        writer.writeheader()
        writer.writerow(completion_row)
    json_path.write_text(json.dumps(completion_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_completion_lines(completion_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    summary = load_diagnostic_summary()
    if not summary:
        print("Text-proxy trial diagnostic summary not found; completion summary not written.")
        return
    completion_row = build_completion_row(summary)
    csv_path, json_path, md_path = write_outputs(completion_row)
    print(
        "Wrote speaker profile text-proxy trial diagnostic completion summary CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile text-proxy trial diagnostic completion summary JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile text-proxy trial diagnostic completion summary note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Queue status: {completion_row['queue_status']}")


if __name__ == "__main__":
    main()
