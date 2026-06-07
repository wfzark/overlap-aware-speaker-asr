from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT
from .evaluate_cer import list_verified_cases


DIAGNOSTIC_COLUMNS = [
    "case_id",
    "best_profile_alignment",
    "profile_confidence_gap",
    "proxy_method",
    "diagnostic_status",
    "diagnostic_note",
]

SUMMARY_COLUMNS = [
    "scope",
    "case_count",
    "swapped_count",
    "direct_count",
    "average_confidence_gap",
    "diagnostic_conclusion",
    "next_method_direction",
]


def load_similarity_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_similarity.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [{key: str(val) for key, val in row.items()} for row in csv.DictReader(f)]


def build_diagnostic_rows(similarity_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    diagnostic_rows: list[dict[str, str]] = []
    for row in similarity_rows:
        case_id = str(row.get("case_id", ""))
        alignment = str(row.get("best_profile_alignment", ""))
        gap = str(row.get("profile_confidence_gap", "0"))
        diagnostic_rows.append(
            {
                "case_id": case_id,
                "best_profile_alignment": alignment,
                "profile_confidence_gap": gap,
                "proxy_method": "text_overlap_profile",
                "diagnostic_status": "text_proxy_diagnostic_complete",
                "diagnostic_note": (
                    f"Text-profile proxy for {case_id} reports {alignment} alignment with gap {gap}; "
                    "this is a risk signal only, not speaker identification."
                ),
            }
        )
    return diagnostic_rows


def build_summary_row(diagnostic_rows: list[dict[str, str]]) -> dict[str, str]:
    if not diagnostic_rows:
        return {
            "scope": "speaker_profile_text_proxy_trial",
            "case_count": "0",
            "swapped_count": "0",
            "direct_count": "0",
            "average_confidence_gap": "0",
            "diagnostic_conclusion": "No similarity rows available.",
            "next_method_direction": "embedding_or_voiceprint_baseline",
        }
    swapped_count = sum(1 for row in diagnostic_rows if row.get("best_profile_alignment") == "swapped")
    direct_count = len(diagnostic_rows) - swapped_count
    avg_gap = round(
        sum(float(row.get("profile_confidence_gap", 0) or 0) for row in diagnostic_rows) / len(diagnostic_rows),
        6,
    )
    conclusion = (
        "All gold cases prefer swapped text-profile alignment; text proxy is useful as a warning sign "
        "but not deployment-ready attribution."
        if swapped_count == len(diagnostic_rows)
        else "Mixed text-profile alignment pattern across gold cases."
    )
    return {
        "scope": "speaker_profile_text_proxy_trial",
        "case_count": str(len(diagnostic_rows)),
        "swapped_count": str(swapped_count),
        "direct_count": str(direct_count),
        "average_confidence_gap": str(avg_gap),
        "diagnostic_conclusion": conclusion,
        "next_method_direction": "embedding_or_voiceprint_baseline",
    }


def build_diagnostic_lines(rows: list[dict[str, str]], summary: dict[str, str]) -> list[str]:
    lines = [
        "# Speaker Profile Text-Proxy Trial Diagnostic",
        "",
        "This generated diagnostic records all-gold text-profile proxy trial results. "
        "It does not claim voiceprint success or improved speaker attribution.",
        "",
        f"Summary: `{summary['swapped_count']}/{summary['case_count']}` cases prefer swapped alignment; "
        f"average confidence gap = {summary['average_confidence_gap']}.",
        "",
        "| case_id | best_profile_alignment | profile_confidence_gap | proxy_method | diagnostic_status | diagnostic_note |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id']} | {row['best_profile_alignment']} | {row['profile_confidence_gap']} | "
            f"{row['proxy_method']} | {row['diagnostic_status']} | {row['diagnostic_note']} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic conclusion",
            "",
            summary["diagnostic_conclusion"],
            "",
            f"Next method direction: `{summary['next_method_direction']}`",
        ]
    )
    return lines


def write_outputs(
    diagnostic_rows: list[dict[str, str]],
    summary_row: dict[str, str],
) -> tuple[Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_text_proxy_trial_diagnostic.csv"
    json_path = tables_dir / "speaker_profile_text_proxy_trial_diagnostic.json"
    md_path = figures_dir / "speaker_profile_text_proxy_trial_diagnostic.md"
    summary_csv_path = tables_dir / "speaker_profile_text_proxy_trial_diagnostic_summary.csv"
    summary_json_path = tables_dir / "speaker_profile_text_proxy_trial_diagnostic_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_COLUMNS)
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    json_path.write_text(json.dumps(diagnostic_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_diagnostic_lines(diagnostic_rows, summary_row)) + "\n", encoding="utf-8")
    with summary_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerow(summary_row)
    summary_json_path.write_text(json.dumps(summary_row, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path, md_path, summary_csv_path, summary_json_path


def main() -> None:
    case_ids = list_verified_cases()
    similarity_rows = load_similarity_rows()
    if not similarity_rows:
        print("Speaker profile similarity table not found; diagnostic not written.")
        return
    missing = [case_id for case_id in case_ids if case_id not in {r["case_id"] for r in similarity_rows}]
    if missing:
        print(f"Warning: missing similarity rows for {missing}")
    diagnostic_rows = build_diagnostic_rows(similarity_rows)
    summary_row = build_summary_row(diagnostic_rows)
    csv_path, json_path, md_path, summary_csv_path, summary_json_path = write_outputs(
        diagnostic_rows, summary_row
    )
    print(f"Wrote speaker profile text-proxy trial diagnostic CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile text-proxy trial diagnostic JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile text-proxy trial diagnostic note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile text-proxy trial diagnostic summary: {summary_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Swapped bias: {summary_row['swapped_count']}/{summary_row['case_count']} cases")


if __name__ == "__main__":
    main()
