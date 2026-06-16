#!/usr/bin/env python3
"""Train and evaluate the Learned Router, writing results to results/tables/.

Usage (from repo root):
    python -m scripts.train_learned_router

Outputs:
    results/tables/learned_router_evaluation.json
    results/tables/learned_router_evaluation.csv
    results/tables/learned_router_tree_rules.txt
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.learned_router import (
    RouterDataset,
    train_router,
    compute_cer_comparison,
    LABEL_TO_METHOD,
)

TABLES_DIR = REPO_ROOT / "results" / "tables"


def main() -> None:
    print("=" * 60)
    print("  Learned Router — Training & Evaluation")
    print("=" * 60)

    cer_csv = TABLES_DIR / "synthetic_split_cer_results.csv"
    routing_csv = TABLES_DIR / "synthetic_split_routing_decisions.csv"
    manifest_csv = TABLES_DIR / "synthetic_split_manifest.csv"

    for path in [cer_csv, routing_csv]:
        if not path.exists():
            print(f"ERROR: required file not found: {path}")
            sys.exit(1)

    # --- Build dataset ---
    print("\n[1/4] Loading dataset ...")
    ds = RouterDataset.from_csvs(cer_csv, routing_csv, manifest_csv)
    train_ds, test_ds = ds.train_test_split()
    print(f"  Total samples: {ds.X.shape[0]}")
    print(f"  Train (dev):   {train_ds.X.shape[0]}")
    print(f"  Test:          {test_ds.X.shape[0]}")
    print(f"  Features:      {ds.X.shape[1]}")

    results_all = {}

    # --- Train logistic regression ---
    print("\n[2/4] Training Logistic Regression ...")
    lr_result = train_router(ds, model_type="logistic_regression")
    print(f"  Train accuracy: {lr_result.train_accuracy:.4f}")
    print(f"  Test accuracy:  {lr_result.test_accuracy:.4f}")
    print(f"\n{lr_result.test_report}")

    lr_cer = compute_cer_comparison(cer_csv, lr_result.predictions, split="test")
    print("  CER comparison (test split):")
    for strategy, avg in lr_cer["average_cer"].items():
        print(f"    {strategy:40s} {avg:.6f}")
    results_all["logistic_regression"] = {
        **lr_result.to_summary_dict(),
        "cer_comparison": lr_cer,
    }

    # --- Train decision tree ---
    print("\n[3/4] Training Decision Tree (max_depth=4) ...")
    dt_result = train_router(ds, model_type="decision_tree", max_depth=4)
    print(f"  Train accuracy: {dt_result.train_accuracy:.4f}")
    print(f"  Test accuracy:  {dt_result.test_accuracy:.4f}")
    print(f"\n{dt_result.test_report}")
    print("  Learned decision rules:")
    print(dt_result.tree_text)

    dt_cer = compute_cer_comparison(cer_csv, dt_result.predictions, split="test")
    print("  CER comparison (test split):")
    for strategy, avg in dt_cer["average_cer"].items():
        print(f"    {strategy:40s} {avg:.6f}")
    results_all["decision_tree"] = {
        **dt_result.to_summary_dict(),
        "cer_comparison": dt_cer,
    }

    # --- Write outputs ---
    print("\n[4/4] Writing results ...")

    # JSON
    out_json = TABLES_DIR / "learned_router_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=2, ensure_ascii=False)
    print(f"  -> {out_json}")

    # CSV summary
    out_csv = TABLES_DIR / "learned_router_evaluation.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "train_accuracy", "test_accuracy",
                         "avg_cer_learned", "avg_cer_oracle", "avg_cer_mixed",
                         "avg_cer_separated", "avg_cer_cleaned"])
        for model_name, data in results_all.items():
            cer_comp = data["cer_comparison"]["average_cer"]
            writer.writerow([
                model_name,
                data["train_accuracy"],
                data["test_accuracy"],
                cer_comp.get("learned_router", ""),
                cer_comp.get("oracle_best", ""),
                cer_comp.get("fixed_mixed_whisper", ""),
                cer_comp.get("fixed_separated_whisper", ""),
                cer_comp.get("fixed_separated_whisper_cleaned", ""),
            ])
    print(f"  -> {out_csv}")

    # Tree rules text
    out_tree = TABLES_DIR / "learned_router_tree_rules.txt"
    with open(out_tree, "w", encoding="utf-8") as f:
        f.write("Decision Tree Rules (max_depth=4)\n")
        f.write("=" * 60 + "\n\n")
        f.write(dt_result.tree_text)
    print(f"  -> {out_tree}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
