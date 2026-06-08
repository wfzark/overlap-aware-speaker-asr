from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


RECEIPT_OPEN_COLUMNS = [
    "case_id",
    "readiness_status",
    "receipt_target",
    "open_action",
    "open_note",
]


def load_readiness_bridge_rows() -> list[dict[str, str]]:
    path = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / "speaker_profile_embedding_trial_execution_receipt_readiness_bridge_checklist.json"
    )
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def build_receipt_open_card_row(readiness_bridge_rows: list[dict[str, str]]) -> dict[str, str]:
    if not readiness_bridge_rows:
        return {}
    head = readiness_bridge_rows[0]
    case_id = str(head.get("case_id", "NoOverlap"))
    readiness_status = str(head.get("readiness_status", "receipt_not_ready"))
    receipt_target = str(head.get("receipt_target", ""))
    return {
        "case_id": case_id,
        "readiness_status": readiness_status,
        "receipt_target": receipt_target,
        "open_action": (
            f"Open {receipt_target} for {case_id} after the speaker-profile receipt-readiness bridge is confirmed."
        ),
        "open_note": (
            f"Execution receipt open card for {case_id} while readiness_status={readiness_status}. "
            "This remains coordination-only and does not itself fill the receipt or claim voiceprint success."
        ),
    }


def build_receipt_open_card_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Speaker Profile Embedding Trial Execution Receipt Open Card",
        "",
        "This generated card gives the next contributor the current speaker-profile execution receipt target to open after the receipt-readiness bridge. "
        "It remains experimental/frontier coordination only and does not fill receipts or claim voiceprint success.",
        "",
        f"- Case id: `{row['case_id']}`",
        f"- Readiness status: `{row['readiness_status']}`",
        f"- Receipt target: `{row['receipt_target']}`",
        f"- Open action: {row['open_action']}",
        f"- Open note: {row['open_note']}",
    ]


def write_outputs(row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_open_card.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_open_card.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_execution_receipt_open_card.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RECEIPT_OPEN_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_receipt_open_card_lines(row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    row = build_receipt_open_card_row(load_readiness_bridge_rows())
    if not row:
        print("Speaker-profile receipt readiness bridge checklist not found; execution receipt open card not written.")
        return
    csv_path, json_path, md_path = write_outputs(row)
    print(f"Wrote speaker profile embedding trial execution receipt open card CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile embedding trial execution receipt open card JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile embedding trial execution receipt open card note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
