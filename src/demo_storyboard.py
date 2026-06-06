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


def write_outputs(cards: list[dict[str, str]]) -> tuple[Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    json_path = tables_dir / "demo_storyboard_cards.json"
    md_path = figures_dir / "demo_storyboard.md"
    json_path.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_demo_storyboard_lines(cards)) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    summary = {
        "baseline": "Selective separation beats blind separation in the current gold benchmark.",
        "cascade": "router_v2 is the balanced default, while cleaned separated output is the robust fallback.",
        "frontier": "Breadth-first artifacts now exist for compute-aware cascade, MeetEval compatibility, speaker profile risk signaling, and qualitative critic notes.",
    }
    cards = build_demo_storyboard_cards(summary)
    json_path, md_path = write_outputs(cards)
    print(f"Wrote demo storyboard cards: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote demo storyboard note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
