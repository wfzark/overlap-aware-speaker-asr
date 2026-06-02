from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged experiments.")
    parser.add_argument("--stage", required=True, choices=["separated", "compare"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(module: str, *extra_args: str) -> None:
    cmd = [sys.executable, "-m", module, *extra_args]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if args.stage == "separated":
        extra = ["--case", "all"]
        if args.overwrite:
            extra.append("--overwrite")
        run("src.transcribe_whisper", *extra, "--mode", "separated")
        merge_args = ["--case", "all"]
        if args.overwrite:
            merge_args.append("--overwrite")
        run("src.merge_speaker_tracks", *merge_args)
        return

    if args.stage == "compare":
        run("src.compare_mixed_vs_separated", "--case", "all")
        return


if __name__ == "__main__":
    main()
