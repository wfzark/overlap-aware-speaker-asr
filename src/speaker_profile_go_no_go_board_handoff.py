from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


HANDOFF_COLUMNS = [
    "handoff_status",
    "overall_state",
    "go_count",
    "checkpoint_count",
    "case_scope",
    "handoff_target",
    "handoff_goal",
    "expected_evidence",
    "handoff_note",
]


def load_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_go_no_go_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_handoff_row(summary: dict[str, str]) -> dict[str, str]:
    overall_state = str(summary.get("overall_state", "execution_not_ready"))
    go_count = str(summary.get("go_count", "0"))
    checkpoint_count = str(summary.get("checkpoint_count", "0"))
    case_scope = str(summary.get("case_scope", "NoOverlap"))
    ready = overall_state == "narrow_execution_ready" and int(go_count or 0) == int(checkpoint_count or 0)
    return {
        "handoff_status": "speaker_profile_go_handoff_ready" if ready else "speaker_profile_go_handoff_pending",
        "overall_state": overall_state,
        "go_count": go_count,
        "checkpoint_count": checkpoint_count,
        "case_scope": case_scope,
        "handoff_target": "results/figures/speaker_profile_embedding_trial_execution_preflight_readiness.md",
        "handoff_goal": (
            f"Advance the narrow embedding-baseline execution preflight for {case_scope} after go-no-go verification."
        ),
        "expected_evidence": "results/tables/speaker_profile_embedding_trial_execution_preflight_readiness.csv",
        "handoff_note": (
            "experimental/frontier speaker-profile go-no-go handoff only; "
            "speaker identification success is not claimed."
        ),
    }


def build_handoff_lines(row: dict[str, str]) -> list[str]:
    return [
        "# Speaker Profile Go-No-Go Board Handoff",
        "",
        "This generated handoff turns the speaker-profile go-no-go board into an embedding trial preflight action. "
        "It does not claim speaker identification success.",
        "",
        "| handoff_status | overall_state | go_count | checkpoint_count | case_scope | handoff_target | handoff_goal | expected_evidence | handoff_note |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
        (
            f"| {row['handoff_status']} | {row['overall_state']} | {row['go_count']} | "
            f"{row['checkpoint_count']} | {row['case_scope']} | {row['handoff_target']} | "
            f"{row['handoff_goal']} | {row['expected_evidence']} | {row['handoff_note']} |"
        ),
    ]


def write_outputs(handoff_row: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_go_no_go_board_handoff.csv"
    json_path = tables_dir / "speaker_profile_go_no_go_board_handoff.json"
    md_path = figures_dir / "speaker_profile_go_no_go_board_handoff.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HANDOFF_COLUMNS)
        writer.writeheader()
        writer.writerow(handoff_row)
    json_path.write_text(json.dumps(handoff_row, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_handoff_lines(handoff_row)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    summary = load_summary()
    if not summary:
        print("Speaker profile go-no-go summary not found; handoff not written.")
        return
    handoff_row = build_handoff_row(summary)
    csv_path, json_path, md_path = write_outputs(handoff_row)
    print(f"Wrote speaker profile go-no-go board handoff CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile go-no-go board handoff JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile go-no-go board handoff note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Handoff status: {handoff_row['handoff_status']}")


if __name__ == "__main__":
    main()
