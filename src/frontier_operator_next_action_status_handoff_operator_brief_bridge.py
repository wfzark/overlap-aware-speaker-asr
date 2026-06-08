from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_COLUMNS = [
    "reentry_frontier",
    "prerequisite_artifact",
    "receipt_target",
    "bridge_note",
]


def load_operator_brief() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_operator_next_action_status_handoff_operator_brief.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_row(operator_brief: dict[str, str]) -> dict[str, str]:
    if not operator_brief:
        return {}
    reentry_frontier = str(operator_brief.get("ready_frontier", "unknown"))
    operator_urgency = str(operator_brief.get("operator_urgency", "queue_status=queue_empty"))
    return {
        "reentry_frontier": reentry_frontier,
        "prerequisite_artifact": "results/figures/frontier_operator_next_action_status_handoff_operator_brief.md",
        "receipt_target": "results/figures/frontier_operator_next_action_status_handoff_runbook_card.md",
        "bridge_note": (
            f"Open the operator brief for {reentry_frontier} first; "
            f"operator_urgency={operator_urgency}. "
            "Then continue through the runbook card. This bridge remains coordination-only and does not claim frontier execution."
        ),
    }


def build_bridge_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Frontier Operator Next-Action Status Handoff Operator Brief Bridge",
        "",
        "This generated bridge connects the status/handoff operator brief to the current runbook card target. "
        "It remains experimental/frontier coordination only and does not claim experiment completion.",
        "",
        "| reentry_frontier | prerequisite_artifact | receipt_target | bridge_note |",
        "| --- | --- | --- | --- |",
        (
            f"| {row['reentry_frontier']} | {row['prerequisite_artifact']} | {row['receipt_target']} | "
            f"{row['bridge_note']} |"
        ),
    ]


def write_outputs(row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_operator_next_action_status_handoff_operator_brief_bridge.csv"
    json_path = tables_dir / "frontier_operator_next_action_status_handoff_operator_brief_bridge.json"
    md_path = figures_dir / "frontier_operator_next_action_status_handoff_operator_brief_bridge.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_lines(row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    row = build_bridge_row(load_operator_brief())
    if not row:
        print("Status handoff operator brief not found; operator brief bridge not written.")
        return
    csv_path, json_path, md_path = write_outputs(row)
    print(
        "Wrote frontier operator next-action status handoff operator brief bridge CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote frontier operator next-action status handoff operator brief bridge JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote frontier operator next-action status handoff operator brief bridge note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
