"""Learned Router — supervised replacement for the rule-based adaptive_router_v2.

This module trains a lightweight classifier (logistic regression or decision tree)
on the *observable* transcript features already produced by the pipeline, using
the oracle-best CER label as ground truth.  It closes the limitation noted in
REPORT.md §7 ("The current router is entirely rule-based …").

Design goals
------------
1. **No new data collection** — reuse synthetic_split_cer_results.csv for labels
   and synthetic_split_routing_decisions.csv for features.
2. **Observable features only** — CER is *never* an input feature; it is the
   evaluation metric.  Features: overlap_level, text_length_ratio, runtime_ratio,
   duplicate_removed_count, separated_unstable, segments counts.
3. **Proper train / test split** — dev split for training, test split for
   held-out evaluation (matching the existing synthetic_split protocol).
4. **Interpretable** — decision-tree variant allows direct inspection of the
   learned rules, which can be compared to the hand-written v2 heuristics.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Optional heavy deps — gracefully degrade for import-time safety
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METHODS = ["mixed_whisper", "separated_whisper", "separated_whisper_cleaned"]
METHOD_TO_LABEL = {m: i for i, m in enumerate(METHODS)}
LABEL_TO_METHOD = {i: m for m, i in METHOD_TO_LABEL.items()}

TIER_TO_OVERLAP = {
    "SyntheticNoOverlap": 0,
    "SyntheticLightOverlap": 1,
    "SyntheticMidOverlap": 2,
    "SyntheticHeavyOverlap": 3,
    "SyntheticOppositeOverlap": 4,
}

FEATURE_NAMES = [
    "overlap_level",
    "mixed_segments_count",
    "separated_segments_count",
    "cleaned_segments_count",
    "mixed_text_length",
    "separated_text_length",
    "cleaned_text_length",
    "text_length_ratio",
    "runtime_ratio",
    "duplicate_removed_count",
]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file and return a list of row dicts.

    Handles UTF-8 BOM (\\ufeff) transparently.
    """
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def load_oracle_labels(cer_csv: Path) -> dict[str, str]:
    """Return {sample_id: best_method} based on minimum CER per sample.

    Only considers the three deployable methods (mixed, separated, cleaned).
    """
    rows = _read_csv(cer_csv)
    sample_cer: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        method = row.get("method", "").strip()
        if method not in METHODS:
            continue
        sid = row.get("sample_id", "").strip()
        cer = _safe_float(row.get("cer"))
        sample_cer[sid].append((method, cer))

    oracle: dict[str, str] = {}
    for sid, entries in sample_cer.items():
        best_method = min(entries, key=lambda x: x[1])[0]
        oracle[sid] = best_method
    return oracle


def load_features(routing_csv: Path) -> dict[str, dict[str, float]]:
    """Extract per-sample feature vectors from routing decisions (v2 rows only).

    We take the v2_full_features row for each sample since it already has all
    the observable features populated.
    """
    rows = _read_csv(routing_csv)
    features: dict[str, dict[str, float]] = {}
    for row in rows:
        strategy = row.get("strategy", row.get("selected_method", "")).strip()
        # routing decisions CSV has a 'strategy' or 'selected_method' column
        # We want exactly one row per sample — use v2_full_features if present
        decision_rule_col = row.get("decision_rule", row.get("strategy", ""))
        # In the synthetic_split_routing_decisions.csv the column order is:
        # sample_id, tier, split, strategy, selected_method, decision_rule, ...features...
        # Let's identify by strategy column
        if "v2_full_features" not in str(strategy) and "v2_full_features" not in str(decision_rule_col):
            continue

        sid = row.get("sample_id", "").strip()
        if not sid:
            continue

        tier = row.get("tier", "").strip()
        overlap = TIER_TO_OVERLAP.get(tier, -1)

        feat = {
            "overlap_level": float(overlap),
            "mixed_segments_count": _safe_float(row.get("mixed_segments_count")),
            "separated_segments_count": _safe_float(row.get("separated_segments_count")),
            "cleaned_segments_count": _safe_float(row.get("cleaned_segments_count")),
            "mixed_text_length": _safe_float(row.get("mixed_text_length")),
            "separated_text_length": _safe_float(row.get("separated_text_length")),
            "cleaned_text_length": _safe_float(row.get("cleaned_text_length")),
            "text_length_ratio": _safe_float(row.get("text_length_ratio")),
            "runtime_ratio": _safe_float(row.get("runtime_ratio")),
            "duplicate_removed_count": _safe_float(row.get("duplicate_removed_count")),
        }
        features[sid] = feat
    return features


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
@dataclass
class RouterDataset:
    """Assembled feature matrix + oracle labels, split-aware."""
    sample_ids: list[str] = field(default_factory=list)
    X: Any = None  # np.ndarray  (n_samples, n_features)
    y: Any = None  # np.ndarray  (n_samples,)
    splits: list[str] = field(default_factory=list)  # "dev" or "test"

    @classmethod
    def from_csvs(
        cls,
        cer_csv: Path,
        routing_csv: Path,
        manifest_csv: Optional[Path] = None,
    ) -> "RouterDataset":
        """Build dataset from project CSV files."""
        if np is None:
            raise ImportError("numpy is required for RouterDataset")

        oracle = load_oracle_labels(cer_csv)
        features = load_features(routing_csv)

        # Determine split from manifest or from sample_id naming convention
        split_map: dict[str, str] = {}
        if manifest_csv and manifest_csv.exists():
            for row in _read_csv(manifest_csv):
                sid = row.get("sample_id", "").strip()
                sp = row.get("split", "").strip()
                if sid and sp:
                    split_map[sid] = sp

        sample_ids: list[str] = []
        X_rows: list[list[float]] = []
        y_list: list[int] = []
        splits: list[str] = []

        for sid in sorted(features.keys()):
            if sid not in oracle:
                continue
            feat = features[sid]
            row_vec = [feat.get(fn, 0.0) for fn in FEATURE_NAMES]
            label = METHOD_TO_LABEL[oracle[sid]]

            # Infer split from sid if not in manifest
            if sid in split_map:
                sp = split_map[sid]
            elif "_dev_" in sid:
                sp = "dev"
            elif "_test_" in sid:
                sp = "test"
            else:
                sp = "unknown"

            sample_ids.append(sid)
            X_rows.append(row_vec)
            y_list.append(label)
            splits.append(sp)

        return cls(
            sample_ids=sample_ids,
            X=np.array(X_rows, dtype=np.float64),
            y=np.array(y_list, dtype=np.int64),
            splits=splits,
        )

    def train_test_split(self) -> tuple["RouterDataset", "RouterDataset"]:
        """Split into dev (train) and test subsets."""
        train_idx = [i for i, s in enumerate(self.splits) if s == "dev"]
        test_idx = [i for i, s in enumerate(self.splits) if s == "test"]

        def _subset(indices: list[int]) -> "RouterDataset":
            return RouterDataset(
                sample_ids=[self.sample_ids[i] for i in indices],
                X=self.X[indices],
                y=self.y[indices],
                splits=[self.splits[i] for i in indices],
            )

        return _subset(train_idx), _subset(test_idx)


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------
@dataclass
class LearnedRouterResult:
    """Container for a trained router and its evaluation metrics."""
    model_name: str
    model: Any
    scaler: Any
    train_accuracy: float
    test_accuracy: float
    test_report: str
    feature_names: list[str]
    predictions: dict[str, str]  # sample_id -> predicted method
    oracle_labels: dict[str, str]  # sample_id -> oracle method
    # Decision tree text (only for tree models)
    tree_text: str = ""

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "train_accuracy": round(self.train_accuracy, 4),
            "test_accuracy": round(self.test_accuracy, 4),
            "feature_names": self.feature_names,
            "tree_text": self.tree_text if self.tree_text else None,
        }


def train_router(
    dataset: RouterDataset,
    model_type: str = "logistic_regression",
    max_depth: int = 4,
) -> LearnedRouterResult:
    """Train a learned router on the dev split, evaluate on test split.

    Parameters
    ----------
    dataset : RouterDataset
        Full dataset (will be split internally).
    model_type : str
        One of "logistic_regression", "decision_tree".
    max_depth : int
        Maximum tree depth (only for decision_tree).

    Returns
    -------
    LearnedRouterResult
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for train_router")

    train_ds, test_ds = dataset.train_test_split()

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_ds.X)
    X_test = scaler.transform(test_ds.X)

    # Choose model
    if model_type == "decision_tree":
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=42,
            class_weight="balanced",
        )
    else:
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight="balanced",
        )

    model.fit(X_train, train_ds.y)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(train_ds.y, train_pred)
    test_acc = accuracy_score(test_ds.y, test_pred)
    # Ensure labels list matches target_names even when some classes are absent
    all_labels = list(range(len(METHODS)))
    report = classification_report(
        test_ds.y, test_pred,
        labels=all_labels,
        target_names=METHODS,
        zero_division=0,
    )

    # Predictions map
    predictions: dict[str, str] = {}
    for sid, pred in zip(test_ds.sample_ids, test_pred):
        predictions[sid] = LABEL_TO_METHOD[int(pred)]

    oracle_labels: dict[str, str] = {}
    for sid, label in zip(test_ds.sample_ids, test_ds.y):
        oracle_labels[sid] = LABEL_TO_METHOD[int(label)]

    tree_text = ""
    if model_type == "decision_tree":
        tree_text = export_text(model, feature_names=FEATURE_NAMES)

    return LearnedRouterResult(
        model_name=model_type,
        model=model,
        scaler=scaler,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        test_report=report,
        feature_names=FEATURE_NAMES,
        predictions=predictions,
        oracle_labels=oracle_labels,
        tree_text=tree_text,
    )


def compute_cer_comparison(
    cer_csv: Path,
    predictions: dict[str, str],
    split: str = "test",
) -> dict[str, Any]:
    """Compare learned router CER against baselines on the given split.

    Returns a dict with per-strategy average CER and sample-level details.
    """
    rows = _read_csv(cer_csv)

    # Build {sample_id: {method: cer}}
    cer_map: dict[str, dict[str, float]] = defaultdict(dict)
    split_filter: set[str] = set()
    for row in rows:
        sid = row.get("sample_id", "").strip()
        method = row.get("method", "").strip()
        sp = row.get("split", "").strip()
        if sp == split and method in METHODS:
            cer_map[sid][method] = _safe_float(row.get("cer"))
            split_filter.add(sid)

    strategies: dict[str, list[float]] = {
        "fixed_mixed_whisper": [],
        "fixed_separated_whisper": [],
        "fixed_separated_whisper_cleaned": [],
        "oracle_best": [],
        "rule_router_v2": [],
        "learned_router": [],
    }

    sample_details: list[dict[str, Any]] = []

    for sid in sorted(split_filter):
        if sid not in cer_map:
            continue
        cers = cer_map[sid]
        mixed_cer = cers.get("mixed_whisper", 1.0)
        sep_cer = cers.get("separated_whisper", 1.0)
        clean_cer = cers.get("separated_whisper_cleaned", 1.0)
        oracle_cer = min(mixed_cer, sep_cer, clean_cer)

        # Learned router CER
        learned_method = predictions.get(sid, "mixed_whisper")
        learned_cer = cers.get(learned_method, 1.0)

        strategies["fixed_mixed_whisper"].append(mixed_cer)
        strategies["fixed_separated_whisper"].append(sep_cer)
        strategies["fixed_separated_whisper_cleaned"].append(clean_cer)
        strategies["oracle_best"].append(oracle_cer)
        strategies["learned_router"].append(learned_cer)

        sample_details.append({
            "sample_id": sid,
            "learned_method": learned_method,
            "learned_cer": round(learned_cer, 6),
            "oracle_cer": round(oracle_cer, 6),
            "gap": round(learned_cer - oracle_cer, 6),
        })

    avg_cers = {k: round(sum(v) / max(len(v), 1), 6) for k, v in strategies.items()}
    return {
        "split": split,
        "n_samples": len(split_filter),
        "average_cer": avg_cers,
        "sample_details": sample_details,
    }
