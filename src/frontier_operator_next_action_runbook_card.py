from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


RUNBOOK_COLUMNS = [
    "recommended_frontier",
    "recommended_action",
    "required_evidence",
    "completion_signal",
    "urgency",
    "runbook_note",
]


def load_operator_brief() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_operator_next_action_operator_brief.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_bridge_rows() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "results" / "tables" / "frontier_operator_next_action_bridge_checklist.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_runbook_card_row(
    operator_brief: dict[str, str],
    bridge_rows: list[dict[str, str]],
) -> dict[str, str]:
    if not operator_brief:
        return {}
    ready_bridge = next((row for row in bridge_rows if row.get("action_lane") == "ready_lane"), {})
    frontier = str(operator_brief.get("ready_frontier", ""))
    target_artifact = str(ready_bridge.get("target_artifact", ""))
    return {
        "recommended_frontier": frontier,
        "recommended_action": str(operator_brief.get("ready_action", "")),
        "required_evidence": str(operator_brief.get("operator_evidence", "")),
        "completion_signal": (
            f"ready_lane verification is complete and the target artifact {target_artifact} is ready to open"
            if target_artifact
            else "ready_lane verification is complete and the target artifact is ready to open"
        ),
        "urgency": str(operator_brief.get("operator_urgency", "")),
        "runbook_note": (
            f"Start with {frontier} as the current ready lane after confirming the bridge checklist. "
            "This remains coordination-only and does not claim frontier completion."
        ),
    }


def build_runbook_card_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Frontier Operator Next-Action Runbook Card",
        "",
        "This generated runbook card condenses the first top-level operator action into a one-page execution card. "
        "It remains experimental/frontier coordination only and does not claim experiment completion.",
        "",
        f"- Recommended frontier: `{row['recommended_frontier']}`",
        f"- Recommended action: `{row['recommended_action']}`",
        f"- Required evidence: `{row['required_evidence']}`",
        f"- Completion signal: `{row['completion_signal']}`",
        f"- Urgency: {row['urgency']}",
        f"- Runbook note: {row['runbook_note']}",
    ]


def write_outputs(row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_operator_next_action_runbook_card.csv"
    json_path = tables_dir / "frontier_operator_next_action_runbook_card.json"
    md_path = figures_dir / "frontier_operator_next_action_runbook_card.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RUNBOOK_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_runbook_card_lines(row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    row = build_runbook_card_row(load_operator_brief(), load_bridge_rows())
    if not row:
        print("Operator brief not found; runbook card not written.")
        return
    csv_path, json_path, md_path = write_outputs(row)
    print(f"Wrote frontier operator next-action runbook card CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier operator next-action runbook card JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier operator next-action runbook card note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
