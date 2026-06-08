from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


OPERATOR_BRIEF_COLUMNS = [
    "ready_frontier",
    "ready_action",
    "ready_target",
    "blocked_frontier",
    "blocked_target",
    "operator_evidence",
    "operator_urgency",
    "operator_note",
]


def load_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_operator_next_action_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_operator_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_operator_next_action_card.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_operator_brief_row(
    summary: dict[str, str],
    operator_rows: list[dict[str, str]],
) -> dict[str, str]:
    if not operator_rows:
        return {}
    ready_row = next((row for row in operator_rows if row.get("action_lane") == "ready_lane"), {})
    blocked_row = next((row for row in operator_rows if row.get("action_lane") == "blocked_lane"), {})
    if not ready_row and not blocked_row:
        return {}
    coordination_state = str(summary.get("coordination_state", "mixed_ready_state"))
    active_lane_count = str(len(operator_rows))
    return {
        "ready_frontier": str(ready_row.get("frontier_name", "")),
        "ready_action": str(ready_row.get("operator_action", "")),
        "ready_target": str(ready_row.get("target_artifact", "")),
        "blocked_frontier": str(blocked_row.get("frontier_name", "")),
        "blocked_target": str(blocked_row.get("target_artifact", "")),
        "operator_evidence": (
            "results/figures/frontier_operator_next_action_card.md; "
            "results/figures/frontier_operator_next_action_bridge_checklist.md"
        ),
        "operator_urgency": f"coordination_state={coordination_state}; active_lanes={active_lane_count}",
        "operator_note": (
            f"Advance {str(ready_row.get('frontier_name', 'the ready lane'))} first while keeping "
            f"{str(blocked_row.get('frontier_name', 'the blocked lane'))} visible as the current unblock target. "
            "This remains coordination-only and does not claim frontier completion."
        ),
    }


def build_operator_brief_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Frontier Operator Next-Action Operator Brief",
        "",
        "This generated brief gives the current top-level frontier operator a plain-language next step summary. "
        "It remains experimental/frontier coordination only and does not claim experiment completion.",
        "",
        f"- Ready frontier: `{row['ready_frontier']}`",
        f"- Ready action: `{row['ready_action']}`",
        f"- Ready target: `{row['ready_target']}`",
        f"- Blocked frontier: `{row['blocked_frontier']}`",
        f"- Blocked target: `{row['blocked_target']}`",
        f"- Evidence path: `{row['operator_evidence']}`",
        f"- Urgency: {row['operator_urgency']}",
        f"- Operator note: {row['operator_note']}",
    ]


def write_outputs(row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_operator_next_action_operator_brief.csv"
    json_path = tables_dir / "frontier_operator_next_action_operator_brief.json"
    md_path = figures_dir / "frontier_operator_next_action_operator_brief.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OPERATOR_BRIEF_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_operator_brief_lines(row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    row = build_operator_brief_row(load_summary(), load_operator_rows())
    if not row:
        print("No operator rows found; operator brief not written.")
        return
    csv_path, json_path, md_path = write_outputs(row)
    print(f"Wrote frontier operator next-action operator brief CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier operator next-action operator brief JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier operator next-action operator brief note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
