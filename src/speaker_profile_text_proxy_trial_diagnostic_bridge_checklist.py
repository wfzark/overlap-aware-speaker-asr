from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "swapped_count",
    "case_count",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_diagnostic_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_text_proxy_trial_diagnostic_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(summary: dict[str, str]) -> list[dict[str, str]]:
    if not summary:
        return []
    swapped_count = str(summary.get("swapped_count", "0"))
    case_count = str(summary.get("case_count", "0"))
    next_method = str(summary.get("next_method_direction", "embedding_or_voiceprint_baseline"))
    return [
        {
            "checklist_order": "1",
            "swapped_count": swapped_count,
            "case_count": case_count,
            "prerequisite_artifact": "results/figures/speaker_profile_text_proxy_trial_diagnostic.md",
            "receipt_target": "results/figures/speaker_profile_embedding_trial_handoff.md",
            "checklist_goal": (
                "Verify all-gold text-proxy diagnostic before opening the embedding trial handoff."
            ),
            "bridge_note": (
                f"Text-proxy diagnostic reports {swapped_count}/{case_count} swapped bias; "
                f"next method direction is {next_method}. "
                "This remains a risk signal only, not speaker identification."
            ),
            "next_gate": "Confirm this bridge before opening the embedding trial handoff target.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Speaker Profile Text-Proxy Trial Diagnostic Bridge Checklist",
        "",
        "This generated checklist connects the text-proxy trial diagnostic to the embedding trial handoff. "
        "It does not claim voiceprint success or improved speaker attribution.",
        "",
        "| checklist_order | swapped_count | case_count | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['swapped_count']} | {row['case_count']} | "
            f"{row['prerequisite_artifact']} | {row['receipt_target']} | {row['checklist_goal']} | "
            f"{row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_text_proxy_trial_diagnostic_bridge_checklist.csv"
    json_path = tables_dir / "speaker_profile_text_proxy_trial_diagnostic_bridge_checklist.json"
    md_path = figures_dir / "speaker_profile_text_proxy_trial_diagnostic_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_diagnostic_summary())
    if not rows:
        print("Text-proxy trial diagnostic summary not found; bridge checklist not written.")
        return
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote speaker profile text-proxy trial diagnostic bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile text-proxy trial diagnostic bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile text-proxy trial diagnostic bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
