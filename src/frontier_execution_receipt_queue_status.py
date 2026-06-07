from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


STATUS_COLUMNS = [
    "scope",
    "meeteval_readiness_status",
    "speaker_profile_readiness_status",
    "external_staging_readiness_status",
    "combined_readiness_status",
    "status_note",
]

READINESS_SOURCES = [
    ("meeteval_readiness_status", "results/tables/meeteval_cpwer_execution_receipt_readiness.json"),
    ("speaker_profile_readiness_status", "results/tables/speaker_profile_embedding_trial_execution_receipt_readiness.json"),
    ("external_staging_readiness_status", "results/tables/external_validation_slice_staging_handoff_receipt_readiness.json"),
]


def load_readiness(path_rel: str) -> dict[str, str]:
    path = PROJECT_ROOT / path_rel
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_status_row(readiness_rows: dict[str, dict[str, str]]) -> dict[str, str]:
    statuses = {
        key: str(row.get("readiness_status", "receipt_not_ready"))
        for key, row in readiness_rows.items()
    }
    all_ready = all(value == "receipt_ready_to_fill" for value in statuses.values())
    combined = "receipt_ready_to_fill" if all_ready and statuses else "receipt_not_ready"
    return {
        "scope": "frontier_execution_receipt_queues",
        "meeteval_readiness_status": statuses.get("meeteval_readiness_status", "receipt_not_ready"),
        "speaker_profile_readiness_status": statuses.get("speaker_profile_readiness_status", "receipt_not_ready"),
        "external_staging_readiness_status": statuses.get("external_staging_readiness_status", "receipt_not_ready"),
        "combined_readiness_status": combined,
        "status_note": (
            "Unified experimental/frontier execution-receipt readiness rollup; "
            "no official benchmark execution or external audio staging is claimed."
        ),
    }


def build_status_lines(row: dict[str, str]) -> list[str]:
    lines = [
        "# Frontier Execution Receipt Queue Status",
        "",
        "This generated note records the unified frontier execution-receipt readiness rollup. "
        "It does not claim benchmark completion.",
        "",
        "| scope | meeteval_readiness_status | speaker_profile_readiness_status | external_staging_readiness_status | combined_readiness_status | status_note |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| {row['scope']} | {row['meeteval_readiness_status']} | {row['speaker_profile_readiness_status']} | "
            f"{row['external_staging_readiness_status']} | {row['combined_readiness_status']} | {row['status_note']} |"
        ),
    ]
    return lines


def write_outputs(status_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "frontier_execution_receipt_queue_status.csv"
    json_path = tables_dir / "frontier_execution_receipt_queue_status.json"
    md_path = figures_dir / "frontier_execution_receipt_queue_status.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_COLUMNS)
        writer.writeheader()
        writer.writerow(status_row)
    json_path.write_text(json.dumps(status_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_status_lines(status_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    readiness_rows = {key: load_readiness(path) for key, path in READINESS_SOURCES}
    status_row = build_status_row(readiness_rows)
    csv_path, json_path, md_path = write_outputs(status_row)
    print(f"Wrote frontier execution receipt queue status CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt queue status JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote frontier execution receipt queue status note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Combined readiness status: {status_row['combined_readiness_status']}")


if __name__ == "__main__":
    main()
