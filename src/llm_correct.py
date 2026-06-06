from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


CSV_COLUMNS = [
    "case_id",
    "label",
    "risk_explanation",
    "candidate_repair",
    "uncertainty_note",
]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return [row for row in csv.DictReader(f) if isinstance(row, dict)]


def build_critic_rows(
    risk_rows: list[dict[str, Any]],
    profile_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    profile_by_case = {str(row.get("case_id", "")): row for row in profile_rows}
    rows: list[dict[str, Any]] = []
    for risk_row in risk_rows:
        case_id = str(risk_row.get("case_id", ""))
        profile_row = profile_by_case.get(case_id, {})
        risk_flags = str(risk_row.get("risk_flags", ""))
        risk_level = str(risk_row.get("risk_level", ""))
        recommended_action = str(risk_row.get("recommended_action", ""))
        best_alignment = str(profile_row.get("best_profile_alignment", ""))
        uncertainty_note = (
            f"Profile alignment still prefers {best_alignment}, so attribution remains uncertain."
            if best_alignment
            else "No profile alignment signal is available, so attribution remains uncertain."
        )
        if risk_flags.strip():
            risk_explanation = f"{risk_flags} suggest unstable separated output and should be treated as a qualitative warning."
        else:
            risk_descriptor = risk_level if risk_level.strip() else "unknown"
            risk_explanation = (
                f"The selector reports a {risk_descriptor} risk state even without explicit flags, so the current transcript still deserves critic review."
            )
        rows.append(
            {
                "case_id": case_id,
                "label": "qualitative/demo",
                "risk_explanation": risk_explanation,
                "candidate_repair": f"Try {recommended_action} before treating the current transcript as final.",
                "uncertainty_note": uncertainty_note,
            }
        )
    return rows


def build_critic_note_lines(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "# LLM Critic Qualitative Note",
        "",
        "This generated note is qualitative only. It uses structured heuristics to imitate a transcript critic and does not claim verified transcript repair.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['case_id']}",
                "",
                f"- Label: `{row['label']}`",
                f"- Risk explanation: {row['risk_explanation']}",
                f"- Candidate repair: {row['candidate_repair']}",
                f"- Uncertainty note: {row['uncertainty_note']}",
                "",
            ]
        )
    return lines


def write_outputs(rows: list[dict[str, Any]]) -> tuple[Path, Path, Path]:
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / "llm_critic_qualitative_summary.csv"
    json_path = tables_dir / "llm_critic_qualitative_summary.json"
    md_path = figures_dir / "llm_critic_qualitative_note.md"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(build_critic_note_lines(rows)) + "\n", encoding="utf-8")
    return csv_path, json_path, md_path


def main() -> None:
    risk_rows = read_csv_rows(PROJECT_ROOT / "results" / "tables" / "risk_aware_selection.csv")
    profile_rows = read_csv_rows(PROJECT_ROOT / "results" / "tables" / "speaker_profile_similarity.csv")
    rows = build_critic_rows(risk_rows, profile_rows)
    csv_path, json_path, md_path = write_outputs(rows)
    print(f"Wrote LLM critic summary: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote LLM critic JSON: {json_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote LLM critic note: {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
