"""ClearFlow Phase 4: rank features and generate the 10 planned feature sets.

All selectors are fitted using the Phase 3 training split only. To keep the
selection phase tractable and reduce majority-class dominance, a reproducible
class-capped reservoir sample is drawn from the training split (up to 10,000
rows per class). The same sample is used for all three selectors.

Methods:
- Random Forest impurity-based feature importance
- One-way ANOVA F-test (the continuous numeric features need not be nonnegative)
- Recursive Feature Elimination (RFE) using a class-weighted Decision Tree

Outputs are written to exports/phase04_feature_selection/.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.tree import DecisionTreeClassifier


TARGET_COLUMN = "Label"
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_MAX_ROWS_PER_CLASS = 10_000
DEFAULT_RANDOM_STATE = 42
DEFAULT_RFE_STEP = 0.20
FEATURE_SET_SIZES = (10, 20, 40)

METHOD_DISPLAY_NAMES = {
    "rf_importance": "Random Forest Importance",
    "anova_f_test": "ANOVA F-test",
    "rfe_decision_tree": "RFE (Decision Tree)",
}


# ---------------------------------------------------------------------------
# Input validation and training-only sample
# ---------------------------------------------------------------------------
def _load_phase03_manifest(project_root: Path) -> dict[str, Any]:
    manifest_path = project_root / "exports" / "phase03_split" / "split_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Phase 3 manifest not found: {manifest_path}. Run Phase 03 first."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    features = manifest.get("model_features")
    if not isinstance(features, list) or not features:
        raise ValueError("Phase 3 manifest must contain a non-empty 'model_features' list.")
    if len(features) != 70:
        raise ValueError(f"Expected 70 model features from Phase 3; found {len(features)}.")
    if len(set(features)) != len(features):
        raise ValueError("Phase 3 manifest contains duplicate model feature names.")
    return manifest


def _reservoir_sample_training_split(
    train_path: Path,
    feature_columns: list[str],
    target_column: str,
    class_order: list[str],
    max_rows_per_class: int,
    chunk_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    """Draw a uniform reservoir sample within each class without loading all rows."""
    rng = np.random.default_rng(random_state)
    class_set = set(class_order)
    reservoirs: dict[str, np.ndarray] = {
        label: np.empty((max_rows_per_class, len(feature_columns)), dtype=np.float32)
        for label in class_order
    }
    stored: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    train_counts: Counter[str] = Counter()
    total_train_rows = 0
    usecols = feature_columns + [target_column]

    print("PASS 1/2: reading the training split and building a class-capped sample...")
    for chunk in pd.read_csv(
        train_path,
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False,
        compression="infer",
        float_precision="round_trip",
    ):
        labels = chunk[target_column].astype("string").str.strip()
        if labels.isna().any():
            raise ValueError("Missing target labels found in the Phase 3 training split.")
        unknown = sorted(set(labels.unique()) - class_set)
        if unknown:
            raise ValueError(f"Unexpected labels in the training split: {unknown}")

        X = chunk[feature_columns].to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(X).all():
            raise ValueError(
                "Non-finite model feature values found in the Phase 3 training split. "
                "Recheck Phase 2 preprocessing."
            )
        y = labels.to_numpy(dtype=str)
        total_train_rows += len(chunk)
        train_counts.update(labels.value_counts().to_dict())

        # Vectorized within-class reservoir updates: only reservoir replacements
        # are written row-by-row into the bounded sample arrays.
        for label in pd.unique(y):
            row_positions = np.flatnonzero(y == label)
            class_rows = X[row_positions]
            seen_before = seen[label]
            stored_before = stored[label]

            fill_count = min(max_rows_per_class - stored_before, len(class_rows))
            if fill_count > 0:
                reservoirs[label][stored_before:stored_before + fill_count] = class_rows[:fill_count]
                stored[label] += fill_count

            remaining_rows = class_rows[fill_count:]
            if len(remaining_rows):
                # For observation index t (zero-based), standard reservoir
                # sampling draws an integer uniformly from [0, t].
                population_indices = np.arange(
                    seen_before + fill_count,
                    seen_before + len(class_rows),
                    dtype=np.int64,
                )
                replacement_indices = rng.integers(0, population_indices + 1)
                replace_mask = replacement_indices < max_rows_per_class
                if replace_mask.any():
                    reservoirs[label][replacement_indices[replace_mask]] = remaining_rows[replace_mask]

            seen[label] += len(class_rows)

    if total_train_rows == 0:
        raise ValueError(f"The Phase 3 training split is empty: {train_path}")

    absent = [label for label in class_order if seen[label] == 0]
    if absent:
        raise ValueError(f"Training split is missing expected classes: {absent}")

    X_parts = [reservoirs[label][:stored[label]] for label in class_order]
    y_parts = [np.full(stored[label], label, dtype=object) for label in class_order]
    X_sample = np.vstack(X_parts).astype(np.float32, copy=False)
    y_sample = np.concatenate(y_parts)

    sample_summary = pd.DataFrame([
        {
            "Label": label,
            "train_rows": int(train_counts[label]),
            "selection_sample_rows": int(stored[label]),
            "sample_pct_of_class": 100.0 * stored[label] / train_counts[label],
            "max_rows_per_class": int(max_rows_per_class),
            "cap_reached": bool(train_counts[label] > max_rows_per_class),
        }
        for label in class_order
    ])

    if not np.isfinite(X_sample).all():
        raise ValueError("The feature-selection sample contains non-finite values.")

    print(f"  Training rows scanned: {total_train_rows:,}")
    print(f"  Feature-selection sample: {len(y_sample):,} rows x {X_sample.shape[1]} features")
    print(f"  Maximum sample size: {max_rows_per_class:,} rows per class")
    return X_sample, y_sample, sample_summary, total_train_rows


# ---------------------------------------------------------------------------
# Feature ranking methods
# ---------------------------------------------------------------------------
def _sort_descending(scores: np.ndarray) -> list[int]:
    """Sort by score descending, preserving Phase 3 feature order for ties."""
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


def _fit_random_forest_rank(
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
    random_state: int,
) -> pd.DataFrame:
    print("Fitting selector 1/3: Random Forest importance...")
    selector = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )
    selector.fit(X, y)
    scores = np.asarray(selector.feature_importances_, dtype=np.float64)
    order = _sort_descending(scores)
    return pd.DataFrame({
        "method": "rf_importance",
        "method_display": METHOD_DISPLAY_NAMES["rf_importance"],
        "feature": [feature_columns[i] for i in order],
        "rank": np.arange(1, len(order) + 1, dtype=int),
        "score": [float(scores[i]) for i in order],
        "score_name": "gini_importance",
        "p_value": np.nan,
        "elimination_round": np.nan,
        "importance_at_elimination": np.nan,
    })


def _fit_anova_rank(
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
) -> pd.DataFrame:
    print("Fitting selector 2/3: ANOVA F-test...")
    # ANOVA F-test accepts continuous values that may be negative. Chi-square
    # feature selection would require an explicit non-negative transformation.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f_scores, p_values = f_classif(X, y)

    finite_scores = f_scores[np.isfinite(f_scores)]
    replacement_for_posinf = (
        float(finite_scores.max()) + 1.0 if len(finite_scores) else 1.0
    )
    scores = np.nan_to_num(
        np.asarray(f_scores, dtype=np.float64),
        nan=-np.inf,
        posinf=replacement_for_posinf,
        neginf=-np.inf,
    )
    p_values = np.nan_to_num(
        np.asarray(p_values, dtype=np.float64),
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    order = _sort_descending(scores)
    return pd.DataFrame({
        "method": "anova_f_test",
        "method_display": METHOD_DISPLAY_NAMES["anova_f_test"],
        "feature": [feature_columns[i] for i in order],
        "rank": np.arange(1, len(order) + 1, dtype=int),
        "score": [float(scores[i]) for i in order],
        "score_name": "anova_f_statistic",
        "p_value": [float(p_values[i]) for i in order],
        "elimination_round": np.nan,
        "importance_at_elimination": np.nan,
    })


def _fit_rfe_rank(
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
    random_state: int,
    step_fraction: float,
) -> pd.DataFrame:
    """Run recursive elimination and create a deterministic full feature order.

    Each round removes the least-important ceil(step_fraction * current_count)
    features. Features surviving to later rounds rank higher. Feature importance
    within an elimination round breaks ties deterministically.
    """
    print("Fitting selector 3/3: recursive feature elimination (Decision Tree)...")
    if not 0 < step_fraction < 1:
        raise ValueError("rfe_step must be strictly between 0 and 1.")

    remaining = list(range(len(feature_columns)))
    elimination_round = np.zeros(len(feature_columns), dtype=int)
    importance_at_elimination = np.zeros(len(feature_columns), dtype=np.float64)
    round_number = 0
    last_importances: dict[int, float] = {}

    while len(remaining) > 1:
        round_number += 1
        selector = DecisionTreeClassifier(
            max_depth=18,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
        )
        selector.fit(X[:, remaining], y)
        importances = np.asarray(selector.feature_importances_, dtype=np.float64)
        for original_index, importance in zip(remaining, importances):
            last_importances[original_index] = float(importance)

        remove_count = min(len(remaining) - 1, max(1, int(math.ceil(step_fraction * len(remaining)))))
        local_order = sorted(
            range(len(remaining)),
            key=lambda local_index: (
                float(importances[local_index]),
                remaining[local_index],
            ),
        )
        remove_local = set(local_order[:remove_count])
        next_remaining: list[int] = []
        for local_index, original_index in enumerate(remaining):
            if local_index in remove_local:
                elimination_round[original_index] = round_number
                importance_at_elimination[original_index] = float(importances[local_index])
            else:
                next_remaining.append(original_index)
        remaining = next_remaining

    # The last remaining feature is ranked ahead of features eliminated earlier.
    final_index = remaining[0]
    elimination_round[final_index] = round_number + 1
    importance_at_elimination[final_index] = last_importances.get(final_index, 0.0)

    order = sorted(
        range(len(feature_columns)),
        key=lambda index: (
            -int(elimination_round[index]),
            -float(importance_at_elimination[index]),
            index,
        ),
    )
    return pd.DataFrame({
        "method": "rfe_decision_tree",
        "method_display": METHOD_DISPLAY_NAMES["rfe_decision_tree"],
        "feature": [feature_columns[i] for i in order],
        "rank": np.arange(1, len(order) + 1, dtype=int),
        "score": [float(importance_at_elimination[i]) for i in order],
        "score_name": "decision_tree_importance_at_elimination",
        "p_value": np.nan,
        "elimination_round": [int(elimination_round[i]) for i in order],
        "importance_at_elimination": [float(importance_at_elimination[i]) for i in order],
    })


# ---------------------------------------------------------------------------
# Feature-set exports
# ---------------------------------------------------------------------------
def _build_feature_sets(
    rankings: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]], pd.DataFrame]:
    sets: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    method_order = ["rf_importance", "anova_f_test", "rfe_decision_tree"]

    for method_index, method in enumerate(method_order, start=1):
        method_ranking = rankings.loc[rankings["method"] == method].sort_values("rank")
        ordered_features = method_ranking["feature"].tolist()
        if len(ordered_features) != len(feature_columns) or len(set(ordered_features)) != len(feature_columns):
            raise RuntimeError(f"{method} did not produce a complete unique feature ranking.")

        method_short = {"rf_importance": "rf", "anova_f_test": "anova", "rfe_decision_tree": "rfe"}[method]
        for size_index, size in enumerate(FEATURE_SET_SIZES, start=1):
            set_id = f"{method_short}_top{size}"
            selected = ordered_features[:size]
            sets[set_id] = {
                "feature_set_id": set_id,
                "feature_set_name": f"Top {size} — {METHOD_DISPLAY_NAMES[method]}",
                "selection_method": method,
                "selection_method_display": METHOD_DISPLAY_NAMES[method],
                "requested_feature_count": int(size),
                "feature_count": len(selected),
                "features": selected,
            }
            summary_rows.append({
                "set_order": (method_index - 1) * len(FEATURE_SET_SIZES) + size_index,
                "feature_set_id": set_id,
                "feature_set_name": sets[set_id]["feature_set_name"],
                "selection_method": method,
                "selection_method_display": METHOD_DISPLAY_NAMES[method],
                "requested_feature_count": int(size),
                "feature_count": len(selected),
            })
            method_ranks = dict(zip(method_ranking["feature"], method_ranking["rank"].astype(int)))
            for feature in selected:
                feature_rows.append({
                    "feature_set_id": set_id,
                    "feature_set_name": sets[set_id]["feature_set_name"],
                    "selection_method": method,
                    "selection_method_display": METHOD_DISPLAY_NAMES[method],
                    "feature_count": len(selected),
                    "feature": feature,
                    "feature_rank": int(method_ranks[feature]),
                })

    all_id = "all_features"
    sets[all_id] = {
        "feature_set_id": all_id,
        "feature_set_name": "All 70 Features (Shared Baseline)",
        "selection_method": "shared_all_features",
        "selection_method_display": "Shared baseline (no feature selection)",
        "requested_feature_count": len(feature_columns),
        "feature_count": len(feature_columns),
        # Keep the original Phase 3 order for the shared all-feature baseline.
        "features": feature_columns.copy(),
    }
    summary_rows.append({
        "set_order": 10,
        "feature_set_id": all_id,
        "feature_set_name": sets[all_id]["feature_set_name"],
        "selection_method": "shared_all_features",
        "selection_method_display": "Shared baseline (no feature selection)",
        "requested_feature_count": len(feature_columns),
        "feature_count": len(feature_columns),
    })
    for rank, feature in enumerate(feature_columns, start=1):
        feature_rows.append({
            "feature_set_id": all_id,
            "feature_set_name": sets[all_id]["feature_set_name"],
            "selection_method": "shared_all_features",
            "selection_method_display": "Shared baseline (no feature selection)",
            "feature_count": len(feature_columns),
            "feature": feature,
            "feature_rank": rank,
        })

    summary = pd.DataFrame(summary_rows).sort_values("set_order").reset_index(drop=True)
    feature_sets_long = pd.DataFrame(feature_rows)

    set_members: dict[str, set[str]] = {
        set_id: set(set_info["features"]) for set_id, set_info in sets.items()
    }
    overlap_rows: list[dict[str, Any]] = []
    ordered_ids = summary["feature_set_id"].tolist()
    for left_id in ordered_ids:
        for right_id in ordered_ids:
            intersection_count = len(set_members[left_id] & set_members[right_id])
            union_count = len(set_members[left_id] | set_members[right_id])
            overlap_rows.append({
                "feature_set_a": left_id,
                "feature_set_a_name": sets[left_id]["feature_set_name"],
                "feature_set_b": right_id,
                "feature_set_b_name": sets[right_id]["feature_set_name"],
                "features_in_a": len(set_members[left_id]),
                "features_in_b": len(set_members[right_id]),
                "intersection_count": intersection_count,
                "union_count": union_count,
                "jaccard_similarity": intersection_count / union_count if union_count else 1.0,
            })

    return summary, feature_sets_long, sets, pd.DataFrame(overlap_rows)


def _make_wide_rankings(rankings: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    wide = pd.DataFrame({"feature": feature_columns})
    suffixes = {
        "rf_importance": {"rank": "rf_rank", "score": "rf_importance_score"},
        "anova_f_test": {"rank": "anova_rank", "score": "anova_f_score", "p_value": "anova_p_value"},
        "rfe_decision_tree": {
            "rank": "rfe_rank",
            "score": "rfe_importance_at_elimination",
            "elimination_round": "rfe_elimination_round",
        },
    }
    for method, renames in suffixes.items():
        subset = rankings.loc[rankings["method"] == method, ["feature", *renames.keys()]].copy()
        subset = subset.rename(columns=renames)
        wide = wide.merge(subset, on="feature", how="left", validate="one_to_one")
    wide["phase03_feature_order"] = np.arange(1, len(feature_columns) + 1)
    return wide.sort_values("phase03_feature_order").reset_index(drop=True)


def run_phase04_feature_selection(
    project_root: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_rows_per_class: int = DEFAULT_MAX_ROWS_PER_CLASS,
    random_state: int = DEFAULT_RANDOM_STATE,
    rfe_step: float = DEFAULT_RFE_STEP,
) -> dict[str, Any]:
    """Run all Phase 4 selectors and write rankings, feature sets, and manifests."""
    project_root = Path(project_root).resolve()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if max_rows_per_class <= 0:
        raise ValueError("max_rows_per_class must be positive.")

    split_manifest = _load_phase03_manifest(project_root)
    train_path = project_root / split_manifest.get("train_output", "data/processed/train.csv.gz")
    if not train_path.is_file():
        raise FileNotFoundError(f"Phase 3 training split not found: {train_path}. Run Phase 03 first.")

    feature_columns = list(split_manifest["model_features"])
    target_column = split_manifest.get("target_column", TARGET_COLUMN)
    class_order = list(split_manifest.get("class_order", []))
    if not class_order:
        raise ValueError("Phase 3 manifest must contain a non-empty 'class_order' list.")

    train_header = pd.read_csv(train_path, nrows=0, compression="infer").columns.tolist()
    missing = sorted(set(feature_columns + [target_column]) - set(train_header))
    if missing:
        raise ValueError(f"Training split is missing expected columns: {missing}")

    output_dir = project_root / "exports" / "phase04_feature_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    X_sample, y_sample, sample_summary, train_rows_scanned = _reservoir_sample_training_split(
        train_path=train_path,
        feature_columns=feature_columns,
        target_column=target_column,
        class_order=class_order,
        max_rows_per_class=max_rows_per_class,
        chunk_size=chunk_size,
        random_state=random_state,
    )

    ranking_frames = [
        _fit_random_forest_rank(X_sample, y_sample, feature_columns, random_state),
        _fit_anova_rank(X_sample, y_sample, feature_columns),
        _fit_rfe_rank(X_sample, y_sample, feature_columns, random_state, rfe_step),
    ]
    rankings = pd.concat(ranking_frames, ignore_index=True)

    set_summary, feature_sets_long, feature_sets, overlap = _build_feature_sets(
        rankings, feature_columns
    )
    rankings_wide = _make_wide_rankings(rankings, feature_columns)

    rankings.to_csv(output_dir / "feature_rankings_long.csv", index=False)
    rankings_wide.to_csv(output_dir / "feature_rankings_wide.csv", index=False)
    feature_sets_long.to_csv(output_dir / "feature_sets_long.csv", index=False)
    set_summary.to_csv(output_dir / "feature_set_summary.csv", index=False)
    overlap.to_csv(output_dir / "feature_set_overlap.csv", index=False)
    sample_summary.to_csv(output_dir / "selection_sample_summary.csv", index=False)

    json_payload = {
        "phase": "04_feature_selection",
        "target_column": target_column,
        "feature_sets": feature_sets,
    }
    with (output_dir / "feature_sets.json").open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2)

    manifest = {
        "phase": "04_feature_selection",
        "input_training_split": str(train_path.relative_to(project_root)),
        "test_split_read": False,
        "test_split_used_for_selection": False,
        "target_column": target_column,
        "input_feature_count": len(feature_columns),
        "input_features": feature_columns,
        "training_rows_scanned": int(train_rows_scanned),
        "selection_sample_rows": int(len(y_sample)),
        "selection_sample_strategy": (
            "Uniform per-class reservoir sample drawn from the training split only; "
            "up to max_rows_per_class rows per class. The same sample is used for all selectors."
        ),
        "max_rows_per_class": int(max_rows_per_class),
        "sample_random_state": int(random_state),
        "rfe_step_fraction": float(rfe_step),
        "methods": [
            {
                "id": "rf_importance",
                "name": METHOD_DISPLAY_NAMES["rf_importance"],
                "implementation": "RandomForestClassifier feature_importances_",
                "parameters": {
                    "n_estimators": 200,
                    "max_features": "sqrt",
                    "min_samples_leaf": 2,
                    "class_weight": "balanced_subsample",
                    "random_state": int(random_state),
                },
            },
            {
                "id": "anova_f_test",
                "name": METHOD_DISPLAY_NAMES["anova_f_test"],
                "implementation": "sklearn.feature_selection.f_classif",
                "notes": "Selected instead of chi-square because features are continuous and need not be non-negative.",
            },
            {
                "id": "rfe_decision_tree",
                "name": METHOD_DISPLAY_NAMES["rfe_decision_tree"],
                "implementation": "Recursive elimination using DecisionTreeClassifier feature_importances_",
                "parameters": {
                    "max_depth": 18,
                    "min_samples_leaf": 5,
                    "class_weight": "balanced",
                    "step_fraction": float(rfe_step),
                    "random_state": int(random_state),
                },
            },
        ],
        "feature_set_sizes": list(FEATURE_SET_SIZES),
        "feature_set_count": int(len(feature_sets)),
        "feature_sets": [
            {
                "feature_set_id": row["feature_set_id"],
                "feature_set_name": row["feature_set_name"],
                "feature_count": int(row["feature_count"]),
                "selection_method": row["selection_method"],
            }
            for row in set_summary.to_dict("records")
        ],
        "outputs": {
            "feature_rankings_long": "exports/phase04_feature_selection/feature_rankings_long.csv",
            "feature_rankings_wide": "exports/phase04_feature_selection/feature_rankings_wide.csv",
            "feature_sets_long": "exports/phase04_feature_selection/feature_sets_long.csv",
            "feature_set_summary": "exports/phase04_feature_selection/feature_set_summary.csv",
            "feature_set_overlap": "exports/phase04_feature_selection/feature_set_overlap.csv",
            "selection_sample_summary": "exports/phase04_feature_selection/selection_sample_summary.csv",
            "feature_sets_json": "exports/phase04_feature_selection/feature_sets.json",
        },
        "caveats": [
            "Selection uses a capped sample of the training split, not every training row.",
            "The capped sample reduces dominance by the largest class and keeps selector memory use bounded.",
            "ANOVA ranks features independently and does not model feature interactions.",
            "Random Forest impurity importance can favor features with many possible split points.",
            "RFE rankings are relative to the Decision Tree estimator and configured elimination step.",
            "The shared all-feature baseline uses Phase 3 feature order and is not method-specific.",
        ],
    }
    with (output_dir / "feature_selection_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    readme = """ClearFlow — Phase 04 Feature Selection exports

Input
- data/processed/train.csv.gz (Phase 03 fixed training split)
- exports/phase03_split/split_manifest.json (feature list, target, class order)
- The test split is never read or used for feature selection.

Methods
1. Random Forest impurity-based feature importance
2. ANOVA F-test (chosen instead of chi-square because numeric features need not be non-negative)
3. RFE with a Decision Tree estimator

Sampling
The three methods use the same reproducible, class-capped reservoir sample from the training split only.
The default cap is 10,000 rows per class (up to 120,000 rows total). Smaller classes are retained in full.
This limits computation and reduces the dominance of the majority class in feature ranking. It is not
model-training class balancing; the training split itself is not changed.

Files
- feature_rankings_long.csv: one row per method-feature ranking (3 x 70 rows)
- feature_rankings_wide.csv: one row per feature with rank/score columns for all methods
- feature_sets_long.csv: feature membership of each of the 10 sets
- feature_set_summary.csv: one row per feature set
- feature_set_overlap.csv: pairwise intersections and Jaccard similarities for the 10 sets
- selection_sample_summary.csv: number of rows sampled from each training class
- feature_sets.json: ordered feature lists for subsequent model training
- feature_selection_manifest.json: parameters, provenance and caveats

Feature sets
- rf_top10, rf_top20, rf_top40
- anova_top10, anova_top20, anova_top40
- rfe_top10, rfe_top20, rfe_top40
- all_features: all 70 features in Phase 3 schema order (shared baseline)

Important
Feature-set score columns have method-specific meanings. Compare ranks and membership, not raw scores
across methods. The selection sample is capped by class; document this when interpreting rankings.
"""
    (output_dir / "README.txt").write_text(readme, encoding="utf-8")

    print("\nFeature-set summary:")
    print(set_summary[["feature_set_id", "feature_count", "selection_method_display"]].to_string(index=False))
    print(f"\nPASS: wrote Phase 4 exports to {output_dir}")

    return {
        "output_dir": output_dir,
        "rankings": rankings,
        "rankings_wide": rankings_wide,
        "feature_set_summary": set_summary,
        "feature_sets_long": feature_sets_long,
        "feature_sets": feature_sets,
        "overlap": overlap,
        "sample_summary": sample_summary,
        "manifest": manifest,
    }
