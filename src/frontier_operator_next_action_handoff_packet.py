from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


HANDOFF_PACKET_COLUMNS = [
    "packet_section",
    "artifact_path",
    "section_role",
]


PACKET_SECTIONS = [
    ("operator_card", "results/figures/frontier_operator_next_action_card.md", "Top-level ready and blocked lane card"),
    ("operator_bridge_checklist", "results/figures/frontier_operator_next_action_bridge_checklist.md", "Verify operator lanes before opening targets"),
    ("operator_brief", "results/figures/frontier_operator_next_action_operator_brief.md", "Plain-language operator summary"),
    ("runbook_card", "results/figures/frontier_operator_next_action_runbook_card.md", "One-page first action execution card"),
    ("frontier_bridge", "results/figures/frontier_operator_next_action_frontier_bridge.md", "Runbook to broader frontier queue alignment"),
    ("frontier_bridge_checklist", "results/figures/frontier_operator_next_action_frontier_bridge_checklist.md", "Verify queue alignment before reopening runbook"),
]


def build_handoff_packet_rows() -> list[dict[str, str]]:
    return [
        {
            "packet_section": section,
            "artifact_path": path,
            "section_role": role,
        }
        for section, path, role in PACKET_SECTIONS
    ]


def build_handoff_packet_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Frontier Operator Next-Action Handoff Packet",
        "",
        "This generated packet consolidates the new top-level frontier operator coordination stack into one entrypoint. "
        "It remains experimental/frontier coordination only and does not claim experiment completion.",
        "",
        "| packet_section | artifact_path | section_role |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['packet_section']} | {row['artifact_path']} | {row['section_role']} |")
    lines.extend(
        [
            "",
            "## Recommended first-open sequence",
            "",
            "1. Open the top-level operator card to confirm the ready and blocked lanes.",
            "2. Verify the operator bridge checklist before following the ready-lane target.",
            "3. Read the operator brief and then the runbook card for the current first frontier (`meeteval_compatibility`).",
            "4. Confirm the frontier bridge and frontier bridge checklist before reopening the runbook card.",
            "",
            "No frontier execution is claimed by this packet alone.",
        ]
    )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_operator_next_action_handoff_packet.csv"
    json_path = tables_dir / "frontier_operator_next_action_handoff_packet.json"
    md_path = figures_dir / "frontier_operator_next_action_handoff_packet.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HANDOFF_PACKET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_handoff_packet_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_handoff_packet_rows()
    csv_path, json_path, md_path = write_outputs(rows)
    print(f"Wrote frontier operator next-action handoff packet CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier operator next-action handoff packet JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier operator next-action handoff packet note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
