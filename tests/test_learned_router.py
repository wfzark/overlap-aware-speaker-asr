"""Tests for src.learned_router — supervised routing module."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to create minimal CSV fixtures
# ---------------------------------------------------------------------------
METHODS = ["mixed_whisper", "separated_whisper", "separated_whisper_cleaned"]


def _make_cer_csv(tmp: Path, rows: list[dict]) -> Path:
    path = tmp / "cer.csv"
    fieldnames = ["sample_id", "tier", "split", "overlap_level", "method",
                   "ref_path", "hyp_path", "ref_chars", "hyp_chars",
                   "edit_distance", "cer"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _make_routing_csv(tmp: Path, rows: list[dict]) -> Path:
    path = tmp / "routing.csv"
    fieldnames = [
        "sample_id", "tier", "split", "strategy", "selected_method",
        "decision_rule",
        "mixed_segments_count", "separated_segments_count",
        "cleaned_segments_count",
        "mixed_text_length", "separated_text_length", "cleaned_text_length",
        "text_length_ratio", "mixed_runtime_sec", "separated_runtime_sec",
        "cleaned_runtime_sec", "runtime_ratio",
        "duplicate_removed_count",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def _sample_cer_row(sid, tier, split, overlap, method, cer):
    return {
        "sample_id": sid, "tier": tier, "split": split,
        "overlap_level": str(overlap), "method": method,
        "ref_path": "", "hyp_path": "",
        "ref_chars": "100", "hyp_chars": "100",
        "edit_distance": str(int(cer * 100)), "cer": str(cer),
    }


def _sample_routing_row(sid, tier, split):
    return {
        "sample_id": sid, "tier": tier, "split": split,
        "strategy": "v2_full_features", "selected_method": "mixed_whisper",
        "decision_rule": "test",
        "mixed_segments_count": "2", "separated_segments_count": "4",
        "cleaned_segments_count": "3",
        "mixed_text_length": "50", "separated_text_length": "120",
        "cleaned_text_length": "80",
        "text_length_ratio": "2.4",
        "mixed_runtime_sec": "1.0", "separated_runtime_sec": "2.0",
        "cleaned_runtime_sec": "2.0",
        "runtime_ratio": "2.0",
        "duplicate_removed_count": "5",
        "notes": "",
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_csvs(tmp_path):
    """Create minimal dev+test CER and routing CSVs (10 dev + 10 test)."""
    cer_rows = []
    routing_rows = []
    tiers = [
        ("SyntheticNoOverlap", 0),
        ("SyntheticLightOverlap", 1),
    ]
    for split_name, idx_range in [("dev", range(1, 6)), ("test", range(1, 6))]:
        for tier_name, overlap in tiers:
            for i in idx_range:
                sid = f"{tier_name}_{split_name}_{i:02d}"
                # mixed is best for NoOverlap, separated is best for Light
                if overlap == 0:
                    cer_rows.append(_sample_cer_row(sid, tier_name, split_name, overlap, "mixed_whisper", 0.05))
                    cer_rows.append(_sample_cer_row(sid, tier_name, split_name, overlap, "separated_whisper", 0.30))
                    cer_rows.append(_sample_cer_row(sid, tier_name, split_name, overlap, "separated_whisper_cleaned", 0.25))
                else:
                    cer_rows.append(_sample_cer_row(sid, tier_name, split_name, overlap, "mixed_whisper", 0.40))
                    cer_rows.append(_sample_cer_row(sid, tier_name, split_name, overlap, "separated_whisper", 0.10))
                    cer_rows.append(_sample_cer_row(sid, tier_name, split_name, overlap, "separated_whisper_cleaned", 0.08))

                routing_rows.append(_sample_routing_row(sid, tier_name, split_name))

    cer_csv = _make_cer_csv(tmp_path, cer_rows)
    routing_csv = _make_routing_csv(tmp_path, routing_rows)
    return cer_csv, routing_csv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestOracleLabels:
    def test_load_oracle_picks_min_cer(self, sample_csvs):
        from src.learned_router import load_oracle_labels
        cer_csv, _ = sample_csvs
        oracle = load_oracle_labels(cer_csv)
        # NoOverlap samples should pick mixed_whisper (cer=0.05)
        assert oracle["SyntheticNoOverlap_dev_01"] == "mixed_whisper"
        # LightOverlap should pick separated_whisper_cleaned (cer=0.08)
        assert oracle["SyntheticLightOverlap_dev_01"] == "separated_whisper_cleaned"

    def test_oracle_returns_all_samples(self, sample_csvs):
        from src.learned_router import load_oracle_labels
        cer_csv, _ = sample_csvs
        oracle = load_oracle_labels(cer_csv)
        assert len(oracle) == 20  # 10 dev + 10 test


class TestLoadFeatures:
    def test_load_features_returns_correct_keys(self, sample_csvs):
        from src.learned_router import load_features, FEATURE_NAMES
        _, routing_csv = sample_csvs
        features = load_features(routing_csv)
        assert len(features) == 20
        for sid, feat in features.items():
            for fn in FEATURE_NAMES:
                assert fn in feat, f"Missing feature {fn} for {sid}"

    def test_feature_values_parsed(self, sample_csvs):
        from src.learned_router import load_features
        _, routing_csv = sample_csvs
        features = load_features(routing_csv)
        f = features["SyntheticNoOverlap_dev_01"]
        assert f["text_length_ratio"] == pytest.approx(2.4)
        assert f["duplicate_removed_count"] == 5.0


class TestRouterDataset:
    def test_from_csvs_shape(self, sample_csvs):
        from src.learned_router import RouterDataset
        cer_csv, routing_csv = sample_csvs
        ds = RouterDataset.from_csvs(cer_csv, routing_csv)
        assert ds.X.shape == (20, 10)  # 20 samples, 10 features
        assert ds.y.shape == (20,)

    def test_train_test_split(self, sample_csvs):
        from src.learned_router import RouterDataset
        cer_csv, routing_csv = sample_csvs
        ds = RouterDataset.from_csvs(cer_csv, routing_csv)
        train, test = ds.train_test_split()
        assert len(train.sample_ids) == 10
        assert len(test.sample_ids) == 10


class TestTrainRouter:
    @pytest.mark.parametrize("model_type", ["logistic_regression", "decision_tree"])
    def test_train_returns_result(self, sample_csvs, model_type):
        from src.learned_router import RouterDataset, train_router
        cer_csv, routing_csv = sample_csvs
        ds = RouterDataset.from_csvs(cer_csv, routing_csv)
        result = train_router(ds, model_type=model_type)
        assert 0.0 <= result.train_accuracy <= 1.0
        assert 0.0 <= result.test_accuracy <= 1.0
        assert len(result.predictions) == 10  # test set size

    def test_decision_tree_has_text(self, sample_csvs):
        from src.learned_router import RouterDataset, train_router
        cer_csv, routing_csv = sample_csvs
        ds = RouterDataset.from_csvs(cer_csv, routing_csv)
        result = train_router(ds, model_type="decision_tree")
        assert len(result.tree_text) > 0
        assert "overlap_level" in result.tree_text or "text_length_ratio" in result.tree_text

    def test_summary_dict(self, sample_csvs):
        from src.learned_router import RouterDataset, train_router
        cer_csv, routing_csv = sample_csvs
        ds = RouterDataset.from_csvs(cer_csv, routing_csv)
        result = train_router(ds, model_type="logistic_regression")
        summary = result.to_summary_dict()
        assert "model_name" in summary
        assert "train_accuracy" in summary
        assert "test_accuracy" in summary


class TestCERComparison:
    def test_comparison_structure(self, sample_csvs):
        from src.learned_router import (
            RouterDataset, train_router, compute_cer_comparison,
        )
        cer_csv, routing_csv = sample_csvs
        ds = RouterDataset.from_csvs(cer_csv, routing_csv)
        result = train_router(ds, model_type="decision_tree")
        comp = compute_cer_comparison(cer_csv, result.predictions, split="test")
        assert comp["split"] == "test"
        assert comp["n_samples"] == 10
        assert "learned_router" in comp["average_cer"]
        assert "oracle_best" in comp["average_cer"]
        # Learned router CER should be >= oracle (can't beat oracle)
        assert comp["average_cer"]["learned_router"] >= comp["average_cer"]["oracle_best"] - 1e-6
