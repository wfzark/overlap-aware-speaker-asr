from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


READINESS_COLUMNS = [
    "scope",
    "handoff_queue_status",
    "scaffold_status",
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
    handoff_completion: dict[str, str],
    scaffold: dict[str, str],
) -> dict[str, str]:
    handoff_queue = str(handoff_completion.get("queue_status", "queue_in_progress"))
    scaffold_status = str(scaffold.get("scaffold_status", "execution_scaffold_only"))
    case_id = str(scaffold.get("case_id", handoff_completion.get("trial_case_target", "NoOverlap")))
    ready = handoff_queue == "queue_complete" and scaffold_status == "execution_scaffold_only"
    return {
        "scope": "speaker_profile_embedding_trial_execution_scaffold",
        "handoff_queue_status": handoff_queue,
        "scaffold_status": scaffold_status,
        "case_id": case_id,
        "readiness_status": "scaffold_ready" if ready else "scaffold_not_ready",
        "readiness_note": (
            "experimental/frontier execution scaffold readiness; "
            "voiceprint or embedding execution is not claimed."
        ),
    }


def build_readiness_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Speaker Profile Embedding Trial Execution Scaffold Readiness",
        "",
        "This generated note records execution scaffold readiness after the embedding trial handoff stack. "
        "It does not claim voiceprint success or improved speaker attribution.",
        "",
        "| scope | handoff_queue_status | scaffold_status | case_id | readiness_status | readiness_note |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| {row['scope']} | {row['handoff_queue_status']} | {row['scaffold_status']} | "
            f"{row['case_id']} | {row['readiness_status']} | {row['readiness_note']} |"
        ),
    ]


def write_outputs(readiness_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_embedding_trial_execution_scaffold_readiness.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_execution_scaffold_readiness.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_execution_scaffold_readiness.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=READINESS_COLUMNS)
        writer.writeheader()
        writer.writerow(readiness_row)
    json_path.write_text(json.dumps(readiness_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_readiness_lines(readiness_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    handoff_completion = load_json_dict(
        "results/tables/speaker_profile_embedding_trial_handoff_completion_summary.json"
    )
    scaffold = load_json_dict("results/tables/speaker_profile_embedding_trial_execution_scaffold.json")
    if not handoff_completion or not scaffold:
        print("Handoff completion or execution scaffold not found; readiness not written.")
        return
    readiness_row = build_readiness_row(handoff_completion, scaffold)
    csv_path, json_path, md_path = write_outputs(readiness_row)
    print(
        "Wrote speaker profile embedding trial execution scaffold readiness CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution scaffold readiness JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution scaffold readiness note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )
    print(f"Readiness status: {readiness_row['readiness_status']}")


if __name__ == "__main__":
    main()
