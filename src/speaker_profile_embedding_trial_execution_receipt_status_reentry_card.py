from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


REENTRY_COLUMNS = [
    "current_case",
    "status_rollup_target",
    "execution_chain_status",
    "reentry_action",
    "reentry_note",
]


def load_status_preflight_rows() -> list[dict[str, str]]:
    path = (
        PROJECT_ROOT
        / "results"
        / "tables"
        / "speaker_profile_embedding_trial_execution_receipt_status_preflight_bridge_checklist.json"
    )
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def load_status_row() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_embedding_trial_execution_status.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_reentry_card_row(
    status_preflight_rows: list[dict[str, str]],
    status_row: dict[str, str],
) -> dict[str, str]:
    if not status_preflight_rows or not status_row:
        return {}
    preflight = status_preflight_rows[0]
    current_case = str(preflight.get("current_case", "unknown"))
    status_target = str(
        preflight.get("receipt_target", "results/figures/speaker_profile_embedding_trial_execution_status.md")
    )
    execution_chain_status = str(status_row.get("execution_chain_status", "execution_chain_in_progress"))
    return {
        "current_case": current_case,
        "status_rollup_target": status_target,
        "execution_chain_status": execution_chain_status,
        "reentry_action": (
            f"After status preflight is confirmed, reopen {status_target} and refresh the speaker-profile status rollup."
        ),
        "reentry_note": (
            f"Status reentry for {current_case} while execution_chain_status={execution_chain_status}. "
            "This remains coordination-only and does not fill the receipt or claim voiceprint success."
        ),
    }


def build_reentry_card_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Speaker Profile Embedding Trial Execution Receipt Status Reentry Card",
        "",
        "This generated card gives the next contributor a one-page reentry instruction after the status preflight bridge. "
        "It remains experimental/frontier coordination only and does not fill receipts or claim voiceprint success.",
        "",
        f"- Current case: `{row['current_case']}`",
        f"- Status rollup target: `{row['status_rollup_target']}`",
        f"- Execution chain status: `{row['execution_chain_status']}`",
        f"- Reentry action: {row['reentry_action']}",
        f"- Reentry note: {row['reentry_note']}",
    ]


def write_outputs(row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_status_reentry_card.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_status_reentry_card.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_execution_receipt_status_reentry_card.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REENTRY_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    json_path.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_reentry_card_lines(row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    row = build_reentry_card_row(load_status_preflight_rows(), load_status_row())
    if not row:
        print("Speaker-profile receipt status preflight or status rollup not found; reentry card not written.")
        return
    csv_path, json_path, md_path = write_outputs(row)
    print(
        "Wrote speaker profile embedding trial execution receipt status reentry card CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution receipt status reentry card JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution receipt status reentry card note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
