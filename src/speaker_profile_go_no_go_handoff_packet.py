from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import PROJECT_ROOT


PACKET_COLUMNS = [
    "packet_order",
    "section_name",
    "artifact_path",
    "section_role",
    "packet_note",
]


PACKET_SECTIONS = [
    (
        "1",
        "go_no_go_board",
        "results/figures/speaker_profile_go_no_go_board.md",
        "Per-checkpoint go/no-go view for the speaker-profile stronger-method chain.",
    ),
    (
        "2",
        "go_no_go_summary",
        "results/figures/speaker_profile_go_no_go_summary.md",
        "Aggregate frontier action line from the go-no-go board.",
    ),
    (
        "3",
        "go_no_go_bridge_checklist",
        "results/figures/speaker_profile_go_no_go_board_bridge_checklist.md",
        "Verify the go-no-go board before reopening embedding trial execution preflight.",
    ),
    (
        "4",
        "go_no_go_handoff",
        "results/figures/speaker_profile_go_no_go_board_handoff.md",
        "Handoff from go-no-go verification to embedding trial execution preflight.",
    ),
    (
        "5",
        "go_no_go_handoff_bridge_checklist",
        "results/figures/speaker_profile_go_no_go_board_handoff_bridge_checklist.md",
        "Verify go-no-go handoff before reopening embedding trial execution scaffold.",
    ),
    (
        "6",
        "go_no_go_handoff_completion_summary",
        "results/figures/speaker_profile_go_no_go_board_handoff_completion_summary.md",
        "Completion rollup for the go-no-go handoff path.",
    ),
    (
        "7",
        "go_no_go_handoff_completion_bridge_checklist",
        "results/figures/speaker_profile_go_no_go_board_handoff_completion_summary_bridge_checklist.md",
        "Verify go-no-go handoff completion before reopening execution scaffold readiness.",
    ),
]


def load_handoff_completion_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_go_no_go_board_handoff_completion_summary.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_packet_rows(summary: dict[str, str]) -> list[dict[str, str]]:
    queue_status = str(summary.get("queue_status", "queue_in_progress"))
    handoff_status = str(summary.get("handoff_status", "speaker_profile_go_handoff_pending"))
    case_scope = str(summary.get("case_scope", "NoOverlap"))
    rows: list[dict[str, str]] = []
    for order, section_name, artifact_path, section_role in PACKET_SECTIONS:
        rows.append(
            {
                "packet_order": order,
                "section_name": section_name,
                "artifact_path": artifact_path,
                "section_role": section_role,
                "packet_note": (
                    f"Speaker-profile go-no-go packet section while handoff_status={handoff_status}, "
                    f"case_scope={case_scope}, queue_status={queue_status}; "
                    "this remains experimental/frontier coordination and does not claim speaker identification success."
                ),
            }
        )
    return rows


def build_packet_lines(rows: list[dict[str, str]], summary: dict[str, str]) -> list[str]:
    queue_status = str(summary.get("queue_status", "queue_in_progress"))
    return [
        "# Speaker Profile Go-No-Go Handoff Packet",
        "",
        "This generated packet provides a compact entrypoint for the speaker-profile go-no-go handoff stack. "
        "It remains experimental/frontier coordination only and does not claim speaker identification success.",
        "",
        f"Current queue status: `{queue_status}`.",
        "",
        "| packet_order | section_name | artifact_path | section_role | packet_note |",
        "| --- | --- | --- | --- | --- |",
        *[
            f"| {row['packet_order']} | {row['section_name']} | {row['artifact_path']} | "
            f"{row['section_role']} | {row['packet_note']} |"
            for row in rows
        ],
    ]


def write_outputs(rows: list[dict[str, str]], summary: dict[str, str]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    csv_path = tables_dir / "speaker_profile_go_no_go_handoff_packet.csv"
    json_path = tables_dir / "speaker_profile_go_no_go_handoff_packet.json"
    md_path = figures_dir / "speaker_profile_go_no_go_handoff_packet.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PACKET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_packet_lines(rows, summary)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    summary = load_handoff_completion_summary()
    if not summary:
        print("Speaker profile go-no-go handoff completion summary not found; handoff packet not written.")
        return
    rows = build_packet_rows(summary)
    csv_path, json_path, md_path = write_outputs(rows, summary)
    print(f"Wrote speaker profile go-no-go handoff packet CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile go-no-go handoff packet JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote speaker profile go-no-go handoff packet note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
