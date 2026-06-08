from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


BRIDGE_CHECKLIST_COLUMNS = [
    "checklist_order",
    "handoff_status",
    "aligned_count",
    "total_count",
    "prerequisite_artifact",
    "receipt_target",
    "checklist_goal",
    "bridge_note",
    "next_gate",
]


def load_handoff() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "meeteval_tokenization_adaptation_handoff.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_bridge_checklist_rows(handoff: dict[str, str]) -> list[dict[str, str]]:
    if not handoff:
        return []
    handoff_status = str(handoff.get("handoff_status", "tokenization_adaptation_handoff_pending"))
    aligned_count = str(handoff.get("aligned_count", "0"))
    total_count = str(handoff.get("total_count", "0"))
    return [
        {
            "checklist_order": "1",
            "handoff_status": handoff_status,
            "aligned_count": aligned_count,
            "total_count": total_count,
            "prerequisite_artifact": "results/figures/meeteval_tokenization_adaptation_handoff.md",
            "receipt_target": "results/figures/frontier_execution_receipt_fill_execution_evidence_receipt.md",
            "checklist_goal": (
                "Verify tokenization adaptation handoff before opening the frontier fill evidence receipt."
            ),
            "bridge_note": (
                f"Tokenization handoff reports handoff_status={handoff_status} with "
                f"{aligned_count}/{total_count} reconciled cases; "
                "confirm before advancing frontier fill execution."
            ),
            "next_gate": "Confirm this bridge before opening the frontier fill evidence receipt.",
        }
    ]


def build_bridge_checklist_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "# MeetEval Tokenization Adaptation Handoff Bridge Checklist",
        "",
        "This generated checklist connects the tokenization adaptation handoff to the frontier fill evidence receipt. "
        "It does not claim full MeetEval benchmark completion.",
        "",
        "| checklist_order | handoff_status | aligned_count | total_count | prerequisite_artifact | receipt_target | checklist_goal | bridge_note | next_gate |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['checklist_order']} | {row['handoff_status']} | {row['aligned_count']} | "
            f"{row['total_count']} | {row['prerequisite_artifact']} | {row['receipt_target']} | "
            f"{row['checklist_goal']} | {row['bridge_note']} | {row['next_gate']} |"
        )
    return lines


def write_outputs(rows: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "meeteval_tokenization_adaptation_handoff_bridge_checklist.csv"
    json_path = tables_dir / "meeteval_tokenization_adaptation_handoff_bridge_checklist.json"
    md_path = figures_dir / "meeteval_tokenization_adaptation_handoff_bridge_checklist.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=BRIDGE_CHECKLIST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_bridge_checklist_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    rows = build_bridge_checklist_rows(load_handoff())
    if not rows:
        print("Tokenization adaptation handoff not found; bridge checklist not written.")
        return
    csv_path, json_path, md_path = write_outputs(rows)
    print(f"Wrote MeetEval tokenization adaptation handoff bridge checklist CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval tokenization adaptation handoff bridge checklist JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote MeetEval tokenization adaptation handoff bridge checklist note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
