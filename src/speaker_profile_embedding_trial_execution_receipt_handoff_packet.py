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
        "receipt_readiness",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_readiness.md",
        "Current readiness rollup for one verified gold case before any receipt fill work begins.",
    ),
    (
        "2",
        "receipt_readiness_bridge_checklist",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_readiness_bridge_checklist.md",
        "Bridge verification for readiness, including swapped-bias context before touching the receipt.",
    ),
    (
        "3",
        "receipt_open_card",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_open_card.md",
        "Single execution receipt target card for the current speaker-profile case.",
    ),
    (
        "4",
        "receipt_open_card_bridge_checklist",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_open_card_bridge_checklist.md",
        "Final gate before reopening the current speaker-profile execution receipt target.",
    ),
    (
        "5",
        "receipt_handoff_packet",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_handoff_packet.md",
        "Compact speaker-profile receipt coordination packet covering the current reentry and execution path.",
    ),
    (
        "6",
        "receipt_handoff_packet_bridge_checklist",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_handoff_packet_bridge_checklist.md",
        "Verify the current speaker-profile receipt packet before reopening the readiness rollup.",
    ),
    (
        "7",
        "receipt_operator_brief",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_operator_brief.md",
        "Plain-language operator summary for the current speaker-profile receipt action.",
    ),
    (
        "8",
        "receipt_operator_brief_bridge",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_operator_brief_bridge.md",
        "Bridge the plain-language operator brief back to the current speaker-profile readiness target.",
    ),
    (
        "9",
        "receipt_operator_brief_bridge_checklist",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_operator_brief_bridge_checklist.md",
        "Verify the operator brief bridge before reopening the current speaker-profile readiness target.",
    ),
    (
        "10",
        "receipt_runbook_card",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_runbook_card.md",
        "One-page execution card for the current speaker-profile receipt action.",
    ),
    (
        "11",
        "receipt_runbook_bridge_checklist",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_runbook_bridge_checklist.md",
        "Verify the current speaker-profile runbook card before reopening the readiness target.",
    ),
    (
        "12",
        "receipt_phase_checkpoint_card",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_phase_checkpoint_card.md",
        "Current explicit completion checkpoint for the speaker-profile receipt runbook path.",
    ),
    (
        "13",
        "receipt_phase_checkpoint_bridge_checklist",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_phase_checkpoint_bridge_checklist.md",
        "Verify the speaker-profile phase checkpoint before reopening the readiness target.",
    ),
    (
        "14",
        "receipt_milestone_card",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_milestone_card.md",
        "Immediate milestone boundary and next unlock for the current speaker-profile receipt path.",
    ),
    (
        "15",
        "receipt_milestone_bridge_checklist",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_milestone_bridge_checklist.md",
        "Verify the speaker-profile milestone card before reopening the readiness target.",
    ),
    (
        "16",
        "receipt_completion_dashboard",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_completion_dashboard.md",
        "One-glance operator-facing dashboard for the current speaker-profile receipt state.",
    ),
    (
        "17",
        "receipt_completion_dashboard_bridge_checklist",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_completion_dashboard_bridge_checklist.md",
        "Verify the speaker-profile completion dashboard before reopening the readiness target.",
    ),
    (
        "18",
        "receipt_status_preflight_bridge_checklist",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_status_preflight_bridge_checklist.md",
        "Verify the completion-dashboard bridge before reopening the speaker-profile status rollup.",
    ),
    (
        "19",
        "receipt_status_reentry_card",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_status_reentry_card.md",
        "One-page reentry instruction for reopening the speaker-profile status rollup.",
    ),
    (
        "20",
        "receipt_status_reentry_bridge_checklist",
        "results/figures/speaker_profile_embedding_trial_execution_receipt_status_reentry_bridge_checklist.md",
        "Verify the speaker-profile status reentry card before reopening the receipt readiness target.",
    ),
]


def load_readiness_summary() -> dict[str, str]:
    path = PROJECT_ROOT / "results" / "tables" / "speaker_profile_embedding_trial_execution_receipt_readiness.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_packet_rows(summary: dict[str, str]) -> list[dict[str, str]]:
    case_id = str(summary.get("case_id", "NoOverlap"))
    readiness_status = str(summary.get("readiness_status", "receipt_not_ready"))
    receipt_template_status = str(summary.get("receipt_template_status", "missing"))
    rows: list[dict[str, str]] = []
    for order, section_name, artifact_path, section_role in PACKET_SECTIONS:
        rows.append(
            {
                "packet_order": order,
                "section_name": section_name,
                "artifact_path": artifact_path,
                "section_role": section_role,
                "packet_note": (
                    f"Speaker-profile execution receipt packet section for {case_id} while "
                    f"readiness_status={readiness_status} and receipt_template_status={receipt_template_status}; "
                    "this remains experimental/frontier coordination only and does not claim voiceprint success."
                ),
            }
        )
    return rows


def build_packet_lines(rows: list[dict[str, str]], summary: dict[str, str]) -> list[str]:
    case_id = str(summary.get("case_id", "NoOverlap"))
    readiness_status = str(summary.get("readiness_status", "receipt_not_ready"))
    return [
        "# Speaker Profile Embedding Trial Execution Receipt Handoff Packet",
        "",
        "This generated note provides a compact entrypoint for the speaker-profile execution receipt sub-stack. "
        "It remains experimental/frontier coordination only and does not fill receipts or claim voiceprint success.",
        "",
        f"Current case: `{case_id}`.",
        f"Current readiness: `readiness_status = {readiness_status}`.",
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

    csv_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_handoff_packet.csv"
    json_path = tables_dir / "speaker_profile_embedding_trial_execution_receipt_handoff_packet.json"
    md_path = figures_dir / "speaker_profile_embedding_trial_execution_receipt_handoff_packet.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=PACKET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_packet_lines(rows, summary)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    summary = load_readiness_summary()
    rows = build_packet_rows(summary)
    csv_path, json_path, md_path = write_outputs(rows, summary)
    print(
        "Wrote speaker profile embedding trial execution receipt handoff packet CSV: "
        f"{csv_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution receipt handoff packet JSON: "
        f"{json_path.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Wrote speaker profile embedding trial execution receipt handoff packet note: "
        f"{md_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
