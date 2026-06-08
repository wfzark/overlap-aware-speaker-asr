from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


PREFLIGHT_READINESS_COLUMNS = [
    "scope",
    "scaffold_queue_status",
    "preflight_pass",
    "case_id",
    "readiness_status",
    "readiness_note",
]


def load_json_dict(path_rel: str) -> dict[str, str]:
    path = PROJECT_ROOT / path_rel
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_readiness_row(
    scaffold_completion: dict[str, str],
    preflight: dict[str, str],
) -> dict[str, str]:
    scaffold_queue = str(scaffold_completion.get("queue_status", "queue_in_progress"))
    preflight_pass = str(preflight.get("preflight_pass", "False"))
    case_id = str(preflight.get("case_id", scaffold_completion.get("case_id", "NoOverlap")))
    ready = scaffold_queue == "queue_complete" and preflight_pass in {"True", "true", True}
    return {
        "scope": "speaker_profile_embedding_trial_execution_preflight",
        "scaffold_queue_status": scaffold_queue,
        "preflight_pass": preflight_pass,
        "case_id": case_id,
        "readiness_status": "preflight_ready" if ready else "preflight_not_ready",
        "readiness_note": (
            "experimental/frontier execution preflight readiness; "
            "voiceprint or embedding model execution is not claimed."
        ),
    }


def build_readiness_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Speaker Profile Embedding Trial Execution Preflight Readiness",
        "",
        "This generated note records execution preflight readiness after scaffold completion. "
        "It does not claim voiceprint success or improved speaker attribution.",
        "",
        "| scope | scaffold_queue_status | preflight_pass | case_id | readiness_status | readiness_note |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| {row['scope']} | {row['scaffold_queue_status']} | {row['preflight_pass']} | "
            f"{row['case_id']} | {row['readiness_status']} | {row['readiness_note']} |"
        ),
    ]


def write_outputs(readiness_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_trial_execution_preflight_readiness.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_execution_preflight_readiness.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_execution_preflight_readiness.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PREFLIGHT_READINESS_COLUMNS)
        writer.writeheader()
        writer.writerow(readiness_row)
    json_path.write_text(json.dumps(readiness_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_readiness_lines(readiness_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    scaffold_completion = load_json_dict(
        "results/tables/speaker_profile_embedding_trial_execution_scaffold_completion_summary.json"
    )
    preflight = load_json_dict("results/tables/speaker_profile_embedding_trial_execution_preflight.json")
    if not scaffold_completion or not preflight:
        print("Scaffold completion or execution preflight not found; readiness not written.")
        return
    readiness_row = build_readiness_row(scaffold_completion, preflight)
    csv_path, json_path, md_path = write_outputs(readiness_row)
    print(
        "Wrote speaker profile embedding trial execution preflight readiness CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution preflight readiness JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution preflight readiness note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Readiness status: {readiness_row['readiness_status']}")


if __name__ == "__main__":
    main()
