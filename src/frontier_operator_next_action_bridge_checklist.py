from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "action_lane",
    "frontier_name",
    "go_no_go_state",
    "prerequisite_artifact",
    "target_artifact",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_operator_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_operator_next_action_card.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_bridge_checklist_rows(operator_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, operator_row in enumerate(operator_rows, start=1):
        action_lane = str(operator_row.get("action_lane", ""))
        frontier_name = str(operator_row.get("frontier_name", "unknown"))
        go_no_go_state = str(operator_row.get("go_no_go_state", ""))
        target_artifact = str(operator_row.get("target_artifact", ""))
        rows.append(
            {
                "checklist_order": str(index),
                "action_lane": action_lane,
                "frontier_name": frontier_name,
                "go_no_go_state": go_no_go_state,
                "prerequisite_artifact": "results/figures/frontier_operator_next_action_card.md",
                "target_artifact": target_artifact,
                "checklist_goal": (
                    f"Verify the operator card lane for {frontier_name} before opening the target artifact."
                ),
                "bridge_note": (
                    f"Operator card reports action_lane={action_lane} and go_no_go_state={go_no_go_state} "
                    f"for {frontier_name}; confirm coordination context before opening the target artifact."
                ),
                "next_gate": f"Confirm this bridge before advancing the {frontier_name} operator lane.",
            }
        )
    return rows


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Frontier Operator Next-Action Bridge Checklist",
        "",
        "This generated checklist turns the top-level frontier operator card into an ordered bridge verification path. "
        "It remains experimental/frontier coordination only and does not claim experiment completion.",
        "",
        "| checklist_order | action_lane | frontier_name | go_no_go_state | prerequisite_artifact | target_artifact | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['action_lane']} | {row['frontier_name']} | "
            f"{row['go_no_go_state']} | {row['prerequisite_artifact']} | {row['target_artifact']} | "
            f"{row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_operator_next_action_bridge_checklist.csv"
    json_path = tables_dir / "frontier_operator_next_action_bridge_checklist.json"
    md_path = figures_dir / "frontier_operator_next_action_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    operator_rows = load_operator_rows()
    rows = build_bridge_checklist_rows(operator_rows)
    csv_path, json_path, md_path = write_outputs(rows)
    print(f"Wrote frontier operator next-action bridge checklist CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier operator next-action bridge checklist JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier operator next-action bridge checklist note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
