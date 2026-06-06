from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


def build_demo_storyboard_cards(summary: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "title": "Problem",
            "body": "Overlap-aware ASR should separate selectively instead of assuming separation always helps.",
        },
        {
            "title": "Pipeline",
            "body": "Mixed ASR, separated ASR, duplicate suppression, adaptive routing, and speaker-aware evaluation form the main decision loop.",
        },
        {
            "title": "Findings",
            "body": f"{summary.get('baseline', '')} {summary.get('cascade', '')}".strip(),
        },
        {
            "title": "Frontier",
            "body": summary.get("frontier", ""),
        },
    ]


def build_demo_storyboard_lines(cards: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Demo Storyboard",
        "",
        "This generated storyboard is a demo-facing summary of the repository's core story and current frontier extensions.",
        "",
        "```mermaid",
        'flowchart LR',
        '    A["Mixed / Separated Audio"] --> B["ASR Paths"]',
        '    B --> C["Router + Risk Selector"]',
        '    C --> D["Speaker-aware / cpCER-lite Evaluation"]',
        '    D --> E["Frontier Extensions"]',
        "```",
        "",
    ]
    for card in cards:
        lines.extend(
            [
                f"## {card['title']}",
                "",
                card["body"],
                "",
            ]
        )
    return lines


def build_demo_walkthrough_steps(summary: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "step_id": "1",
            "focus": "Problem framing",
            "talk_track": "Start with the core question: overlap exists, but separation should be selective rather than automatic.",
            "artifact_anchor": "README.md",
        },
        {
            "step_id": "2",
            "focus": "Baseline evidence",
            "talk_track": summary.get("baseline", ""),
            "artifact_anchor": "REPORT.md",
        },
        {
            "step_id": "3",
            "focus": "Routing takeaway",
            "talk_track": summary.get("router", ""),
            "artifact_anchor": "results/figures/compute_aware_cascade_summary.md",
        },
        {
            "step_id": "4",
            "focus": "Frontier breadth",
            "talk_track": summary.get("frontier", ""),
            "artifact_anchor": "results/figures/project_harness_report.md",
        },
        {
            "step_id": "5",
            "focus": "Next-step framing",
            "talk_track": "Close by showing that the next work is structured: benchmark handoff, external prioritization, and qualitative repair paths are already staged.",
            "artifact_anchor": "docs/roadmap.md",
        },
    ]


def build_demo_walkthrough_lines(steps: list[dict[str, str]]) -> list[str]:
    lines = [
        "# Demo Walkthrough",
        "",
        "This generated walkthrough suggests a short demo sequence for explaining the repository to a new viewer.",
        "",
        "| step_id | focus | talk_track | artifact_anchor |",
        "| --- | --- | --- | --- |",
    ]
    for step in steps:
        lines.append(
            f"| {step['step_id']} | {step['focus']} | {step['talk_track']} | {step['artifact_anchor']} |"
        )
    return lines


def write_outputs(cards: list[dict[str, str]], steps: list[dict[str, str]]) -> tuple[Path, Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    json_path = tables_dir / "demo_storyboard_cards.json"
    md_path = figures_dir / "demo_storyboard.md"
    walkthrough_json_path = tables_dir / "demo_walkthrough_steps.json"
    walkthrough_md_path = figures_dir / "demo_walkthrough.md"
    json_path.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_demo_storyboard_lines(cards)) + "\n", encoding="utf-8")
    walkthrough_json_path.write_text(json.dumps(steps, ensure_ascii=False, indent=2), encoding="utf-8")
    walkthrough_md_path.write_text("\n".join(build_demo_walkthrough_lines(steps)) + "\n", encoding="utf-8")
    return json_path, md_path, walkthrough_json_path, walkthrough_md_path


def main() -> None:
    summary = {
        "baseline": "Selective separation beats blind separation in the current gold benchmark.",
        "cascade": "router_v2 is the balanced default, while cleaned separated output is the robust fallback.",
        "router": "router_v2 matches oracle-best average CER, and the compute-aware frontier turns that into deployment-facing tradeoff guidance.",
        "frontier": "Breadth-first artifacts now cover compute-aware cascade, MeetEval compatibility, speaker profile risk signaling, qualitative critics, and external prioritization.",
    }
    cards = build_demo_storyboard_cards(summary)
    steps = build_demo_walkthrough_steps(summary)
    json_path, md_path, walkthrough_json_path, walkthrough_md_path = write_outputs(cards, steps)
    print(f"Wrote demo storyboard cards: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard note: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough steps: {walkthrough_json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo walkthrough note: {walkthrough_md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
