from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


CSV_COLUMNS = [
    "dataset_name",
    "label",
    "source_note",
    "license_note",
    "fit_note",
    "first_preprocessing_step",
    "next_action",
]

PRIORITIZATION_COLUMNS = [
    "dataset_name",
    "label",
    "priority_tier",
    "recommended_order",
    "readiness_note",
    "why_now",
    "next_action",
]


def build_external_validation_candidate_rows() -> list[dict[str, str]]:
    return [
        {
            "dataset_name": "AISHELL-4",
            "label": "external/sanity-check",
            "source_note": "Official AISHELL-4 release page and paper.",
            "license_note": "Check the official license terms before local reuse.",
            "fit_note": "Chinese multi-speaker meeting data with realistic overlap and strong domain relevance.",
            "first_preprocessing_step": "Map a tiny subset into the repository speaker-reference format.",
            "next_action": "Confirm license and choose a tiny sanity-check slice.",
        },
        {
            "dataset_name": "AliMeeting",
            "label": "external/sanity-check",
            "source_note": "Official AliMeeting dataset release and paper.",
            "license_note": "Check the official license terms before local reuse.",
            "fit_note": "Meeting-style overlap and diarization structure are close to the current ASR framing.",
            "first_preprocessing_step": "Select one short meeting excerpt and normalize segment timestamps.",
            "next_action": "Confirm license and choose one compact overlap-heavy excerpt.",
        },
        {
            "dataset_name": "AMI",
            "label": "external/sanity-check",
            "source_note": "AMI Meeting Corpus distribution page.",
            "license_note": "Check the AMI corpus license and redistribution rules before use.",
            "fit_note": "Classic meeting benchmark with overlap and established evaluation conventions.",
            "first_preprocessing_step": "Extract one short meeting clip and align speaker annotations to repo schema.",
            "next_action": "Confirm license and compare one clip against current meeting-style exports.",
        },
        {
            "dataset_name": "LibriCSS",
            "label": "external/sanity-check",
            "source_note": "LibriCSS release page and benchmark paper.",
            "license_note": "Check the official license terms before local reuse.",
            "fit_note": "Overlap-heavy conversational speech is useful for a focused external overlap sanity-check.",
            "first_preprocessing_step": "Map one overlap condition into the repository transcript/reference format.",
            "next_action": "Confirm license and choose one overlap condition for a narrow sanity-check.",
        },
    ]


def build_external_validation_candidate_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# External Validation Candidates",
        "",
        "This generated note lists candidate external sanity-check datasets. It does not claim that any external benchmark has already been run.",
        "",
        "| dataset_name | label | source_note | license_note | fit_note | first_preprocessing_step | next_action |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset_name']} | {row['label']} | {row['source_note']} | {row['license_note']} | "
            f"{row['fit_note']} | {row['first_preprocessing_step']} | {row['next_action']} |"
        )
    return lines


def build_external_validation_prioritization_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    by_name = {row["dataset_name"]: row for row in rows}
    ordered_names = ["AISHELL-4", "AliMeeting", "AMI", "LibriCSS"]
    priority_map = {
        "AISHELL-4": {
            "priority_tier": "start_here",
            "recommended_order": "1",
            "readiness_note": "License check plus a tiny repo-format mapping are still required before use.",
            "why_now": "Chinese meeting overlap and domain fit make this the closest first sanity-check target.",
        },
        "AliMeeting": {
            "priority_tier": "near_term_backup",
            "recommended_order": "2",
            "readiness_note": "License confirmation and timestamp normalization are still required before use.",
            "why_now": "Meeting-style structure stays close to the current framing if AISHELL-4 access is inconvenient.",
        },
        "AMI": {
            "priority_tier": "cross_domain_reference",
            "recommended_order": "3",
            "readiness_note": "AMI license and redistribution rules should be checked before any local slice is staged.",
            "why_now": "Classic benchmark value is high, but the domain is a looser fit than the Chinese meeting candidates.",
        },
        "LibriCSS": {
            "priority_tier": "specialized_followup",
            "recommended_order": "4",
            "readiness_note": "License check and overlap-condition selection are still required before use.",
            "why_now": "This is strongest for overlap stress-testing after one meeting-style sanity-check path is in place.",
        },
    }
    prioritized_rows: list[dict[str, str]] = []
    for dataset_name in ordered_names:
        base_row = by_name.get(dataset_name)
        if base_row is None:
            continue
        priority = priority_map[dataset_name]
        prioritized_rows.append(
            {
                "dataset_name": dataset_name,
                "label": base_row["label"],
                "priority_tier": priority["priority_tier"],
                "recommended_order": priority["recommended_order"],
                "readiness_note": priority["readiness_note"],
                "why_now": priority["why_now"],
                "next_action": base_row["next_action"],
            }
        )
    return prioritized_rows


def build_external_validation_prioritization_lines(
    rows: list[dict[str, str]],
) -> list[str]:
    lines = [
        "# External Validation Prioritization",
        "",
        "This generated note recommends which external sanity-check candidate should be tried first. It does not claim that any external benchmark has already been run.",
        "",
        "| dataset_name | label | priority_tier | recommended_order | readiness_note | why_now | next_action |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset_name']} | {row['label']} | {row['priority_tier']} | {row['recommended_order']} | "
            f"{row['readiness_note']} | {row['why_now']} | {row['next_action']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "external_validation_candidates.csv"
    json_path = tables_dir / "external_validation_candidates.json"
    md_path = figures_dir / "external_validation_candidates.md"
    prioritization_rows = build_external_validation_prioritization_rows(rows)
    prioritization_csv_path = tables_dir / "external_validation_prioritization.csv"
    prioritization_json_path = tables_dir / "external_validation_prioritization.json"
    prioritization_md_path = figures_dir / "external_validation_prioritization.md"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_external_validation_candidate_lines(rows)) + "\n", encoding="utf-8")
    with prioritization_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PRIORITIZATION_COLUMNS)
        writer.writeheader()
        writer.writerows(prioritization_rows)
    prioritization_json_path.write_text(json.dumps(prioritization_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    prioritization_md_path.write_text(
        "\n".join(build_external_validation_prioritization_lines(prioritization_rows)) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path, md_path, prioritization_csv_path, prioritization_json_path, prioritization_md_path


def main() -> None:
    rows = build_external_validation_candidate_rows()
    csv_path, json_path, md_path, prioritization_csv_path, prioritization_json_path, prioritization_md_path = write_outputs(rows)
    print(f"Wrote external validation candidates: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote external validation JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote external validation note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote external validation prioritization: {prioritization_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote external validation prioritization JSON: {prioritization_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote external validation prioritization note: {prioritization_md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
