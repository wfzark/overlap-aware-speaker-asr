from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


DASHBOARD_COLUMNS = [
    "current_first_frontier",
    "blocked_frontier",
    "next_milestone",
    "remaining_frontier_count",
    "dominant_blocker",
    "dashboard_note",
]


def load_operator_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_operator_next_action_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_milestone_card() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_operator_next_action_milestone_card.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_dashboard_row(
    summary: dict[str, str],
    milestone: dict[str, str],
) -> dict[str, str]:
    ready_frontier = str(summary.get("ready_frontier", ""))
    blocked_frontier = str(summary.get("blocked_frontier", ""))
    next_milestone = str(milestone.get("next_milestone", ""))
    remaining = str(milestone.get("remaining_frontier_count", "0"))
    return {
        "current_first_frontier": ready_frontier,
        "blocked_frontier": blocked_frontier,
        "next_milestone": next_milestone,
        "remaining_frontier_count": remaining,
        "dominant_blocker": blocked_frontier or "none",
        "dashboard_note": (
            f"{ready_frontier or 'The ready frontier'} leads the top-level operator chain while "
            f"{blocked_frontier or 'no blocked frontier'} remains the dominant coordination blocker. "
            "This remains coordination-only and does not claim frontier execution."
        ),
    }


def build_dashboard_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Frontier Operator Next-Action Completion Dashboard",
        "",
        "This generated dashboard summarizes the current top-level operator chain at a glance. "
        "It remains experimental/frontier coordination only and does not claim experiment completion.",
        "",
        f"- Current first frontier: `{row['current_first_frontier']}`",
        f"- Blocked frontier: `{row['blocked_frontier']}`",
        f"- Next milestone: `{row['next_milestone']}`",
        f"- Remaining frontier count after milestone: `{row['remaining_frontier_count']}`",
        f"- Dominant blocker: `{row['dominant_blocker']}`",
        f"- Dashboard note: {row['dashboard_note']}",
    ]


def write_outputs(row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_operator_next_action_completion_dashboard.csv"
    json_path = tables_dir / "frontier_operator_next_action_completion_dashboard.json"
    md_path = figures_dir / "frontier_operator_next_action_completion_dashboard.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=DASHBOARD_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_dashboard_lines(row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    row = build_dashboard_row(load_operator_summary(), load_milestone_card())
    csv_path, json_path, md_path = write_outputs(row)
    print(f"Wrote frontier operator next-action completion dashboard CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier operator next-action completion dashboard JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier operator next-action completion dashboard note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
