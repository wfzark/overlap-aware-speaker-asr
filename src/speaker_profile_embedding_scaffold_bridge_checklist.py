from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "scaffold_status",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_embedding_scaffold() -> dict[str, str]:
    scaffold_path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_embedding_scaffold.json"
    if not scaffold_path.exists():
        return {}
    payload = json.loads(scaffold_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(scaffold: dict[str, str]) -> list[dict[str, str]]:
    scaffold_status = str(scaffold.get("scaffold_status", "scaffold_only"))
    method_direction = str(scaffold.get("method_direction", "embedding_or_voiceprint_baseline"))
    return [
        {
            "checklist_order": "1",
            "scaffold_status": scaffold_status,
            "prerequisite_artifact": "results/figures/speaker_profile_embedding_scaffold.md",
            "receipt_target": "results/figures/speaker_profile_method_receipt.md",
            "checklist_goal": (
                f"Verify the embedding scaffold bridge for {method_direction} before any stronger-method trial claim."
            ),
            "bridge_note": (
                f"Scaffold remains {scaffold_status}; confirm the stronger-method direction before opening the method receipt target."
            ),
            "next_gate": "Confirm this bridge before opening the speaker profile method receipt target.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Speaker Profile Embedding Scaffold Bridge Checklist",
        "",
        "This generated checklist turns the embedding scaffold into a row-by-row bridge verification path. "
        "It remains diagnostic-only and does not claim improved speaker attribution.",
        "",
        "| checklist_order | scaffold_status | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['scaffold_status']} | {row['prerequisite_artifact']} | "
            f"{row['receipt_target']} | {row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_scaffold_bridge_checklist.csv"
    json_path = tables_dir / "speaker_profile_embedding_scaffold_bridge_checklist.json"
    md_path = figures_dir / "speaker_profile_embedding_scaffold_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    scaffold = load_embedding_scaffold()
    rows = build_bridge_checklist_rows(scaffold)
    csv_path, json_path, md_path = write_outputs(rows)
    print(
        "Wrote speaker profile embedding scaffold bridge checklist CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding scaffold bridge checklist JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding scaffold bridge checklist note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
