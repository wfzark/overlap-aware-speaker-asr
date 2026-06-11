from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(*parts: str) -> str:
    """Resolve a repository-relative path."""
    return str(PROJECT_ROOT.joinpath(*parts))


def load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path.relative_to(PROJECT_ROOT)}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_audio_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    return list(config.get("audio_cases", []))


def main() -> None:
    config = load_config()
    for case in get_audio_cases(config):
        print(f"{case['id']}: {case['mixed']} (overlap_level={case['overlap_level']})")


if __name__ == "__main__":
    main()
