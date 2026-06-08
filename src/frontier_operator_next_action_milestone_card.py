from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


MILESTONE_COLUMNS = [
    "next_milestone",
    "unlocks",
    "remaining_frontier_count",
    "milestone_note",
]


def load_operator_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_operator_next_action_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_milestone_card_row(summary: dict[str, str]) -> dict[str, str]:
    ready_frontier = str(summary.get("ready_frontier", ""))
    blocked_frontier = str(summary.get("blocked_frontier", ""))
    return {
        "next_milestone": "ready_lane_checkpoint_complete",
        "unlocks": (
            f"Advance coordination focus to {blocked_frontier} after the current ready-lane checkpoint closes"
            if blocked_frontier
            else "Advance coordination focus to the next frontier after the current ready-lane checkpoint closes"
        ),
        "remaining_frontier_count": "1" if blocked_frontier else "0",
        "milestone_note": (
            f"Completing the current {ready_frontier or 'ready-lane'} checkpoint closes the first top-level operator milestone; "
            "this remains coordination-only and does not claim frontier execution."
        ),
    }


def build_milestone_card_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Frontier Operator Next-Action Milestone Card",
        "",
        "This generated milestone card shows the immediate completion boundary for the top-level operator chain. "
        "It remains experimental/frontier coordination only and does not claim experiment completion.",
        "",
        f"- Next milestone: `{row['next_milestone']}`",
        f"- Unlocks: {row['unlocks']}",
        f"- Remaining frontier count after milestone: `{row['remaining_frontier_count']}`",
        f"- Milestone note: {row['milestone_note']}",
    ]


def write_outputs(row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_operator_next_action_milestone_card.csv"
    json_path = tables_dir / "frontier_operator_next_action_milestone_card.json"
    md_path = figures_dir / "frontier_operator_next_action_milestone_card.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=MILESTONE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_milestone_card_lines(row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    row = build_milestone_card_row(load_operator_summary())
    csv_path, json_path, md_path = write_outputs(row)
    print(f"Wrote frontier operator next-action milestone card CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier operator next-action milestone card JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier operator next-action milestone card note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
