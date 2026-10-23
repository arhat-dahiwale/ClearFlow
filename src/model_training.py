
"""ClearFlow Phase 05: train and evaluate the 200 model configurations.

Protocol:
- Reuse the fixed Phase 03 train/test split and Phase 04 feature sets.
- Use one fixed, class-capped reservoir sample from the training split only.
- Fit all preprocessing on training data only.
- Keep the full test split untouched for evaluation.
- Save each fitted model for the later SHAP phase.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight


TARGET_COLUMN = "Label"
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_MAX_ROWS_PER_CLASS = 10_000
DEFAULT_SAMPLE_RANDOM_STATE = 42
DEFAULT_SEEDS = (42, 43, 44, 45, 46)

MODEL_ORDER = (
    "decision_tree",
    "random_forest",
    "xgboost",
    "logistic_regression",
)

MODEL_DISPLAY_NAMES = {
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write through a temporary file to reduce partial-checkpoint risk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


def _load_feature_sets(
    path: Path,
    expected_features: list[str],
) -> dict[str, dict[str, Any]]:
    """Load and validate the exact 10 feature sets from Phase 04."""
    payload = _read_json(path)
    feature_sets = payload.get("feature_sets")

    if not isinstance(feature_sets, dict) or len(feature_sets) != 10:
        raise ValueError(
            "Phase 04 feature_sets.json must contain exactly 10 feature sets."
        )

    expected_sizes = {
        "rf_top10": 10,
        "rf_top20": 20,
        "rf_top40": 40,
        "anova_top10": 10,
        "anova_top20": 20,
        "anova_top40": 40,
        "rfe_top10": 10,
        "rfe_top20": 20,
        "rfe_top40": 40,
        "all_features": len(expected_features),
    }

    if set(feature_sets) != set(expected_sizes):
        raise ValueError(
            "Unexpected feature-set IDs in Phase 04 JSON. "
            f"Expected {sorted(expected_sizes)}; "
            f"found {sorted(feature_sets)}"
        )

    expected_feature_set = set(expected_features)

    for set_id, expected_size in expected_sizes.items():
        features = feature_sets[set_id].get("features")

        if not isinstance(features, list) or len(features) != expected_size:
            raise ValueError(
                f"Feature set {set_id!r} should contain "
                f"{expected_size} features."
            )

        if len(set(features)) != len(features):
            raise ValueError(
                f"Feature set {set_id!r} contains duplicate features."
            )

        unknown = set(features) - expected_feature_set
        if unknown:
            raise ValueError(
                f"Feature set {set_id!r} contains unknown features: "
                f"{sorted(unknown)}"
            )

    if feature_sets["all_features"]["features"] != expected_features:
        raise ValueError(
            "The all_features set must preserve the Phase 03 feature order."
        )

    return feature_sets


def _make_training_sample(
    train_path: Path,
    feature_columns: list[str],
    target_column: str,
    class_order: list[str],
    max_rows_per_class: int,
    chunk_size: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, int]:
    """Make one reproducible uniform reservoir sample from training rows."""
    if max_rows_per_class <= 0 or chunk_size <= 0:
        raise ValueError("max_rows_per_class and chunk_size must be positive.")

    rng = np.random.default_rng(random_state)
    class_set = set(class_order)

    reservoirs = {
        label: np.empty(
            (max_rows_per_class, len(feature_columns)),
            dtype=np.float32,
        )
        for label in class_order
    }

    stored: Counter[str] = Counter()
    seen: Counter[str] = Counter()
    train_counts: Counter[str] = Counter()
    total_rows = 0

    dtype_map = {feature: np.float32 for feature in feature_columns}
    dtype_map[target_column] = "string"

    print("PASS 1/3: creating the fixed class-capped training sample...")

    for chunk in pd.read_csv(
        train_path,
        usecols=feature_columns + [target_column],
        chunksize=chunk_size,
        compression="infer",
        low_memory=False,
        dtype=dtype_map,
    ):
        labels = chunk[target_column].astype("string").str.strip()

        if labels.isna().any():
            raise ValueError(
                "Missing target labels found in the training split."
            )

        unknown = sorted(set(labels.unique()) - class_set)
        if unknown:
            raise ValueError(
                f"Unexpected labels in training split: {unknown}"
            )

        X_chunk = chunk[feature_columns].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        if not np.isfinite(X_chunk).all():
            raise ValueError(
                "Non-finite model features found in the training split."
            )

        y_chunk = labels.to_numpy(dtype=str)
        total_rows += len(chunk)
        train_counts.update(labels.value_counts().to_dict())

        for label in pd.unique(y_chunk):
            positions = np.flatnonzero(y_chunk == label)
            class_rows = X_chunk[positions]

            seen_before = seen[label]
            stored_before = stored[label]

            fill_count = min(
                max_rows_per_class - stored_before,
                len(class_rows),
            )

            if fill_count > 0:
                reservoirs[label][
                    stored_before:stored_before + fill_count
                ] = class_rows[:fill_count]

                stored[label] += fill_count

            remaining_rows = class_rows[fill_count:]

            if len(remaining_rows):
                population_indices = np.arange(
                    seen_before + fill_count,
                    seen_before + len(class_rows),
                    dtype=np.int64,
                )

                replacement_indices = rng.integers(
                    0,
                    population_indices + 1,
                )

                replace_mask = (
                    replacement_indices < max_rows_per_class
                )

                if replace_mask.any():
                    reservoirs[label][
                        replacement_indices[replace_mask]
                    ] = remaining_rows[replace_mask]

            seen[label] += len(class_rows)

    absent = [
        label for label in class_order
        if seen[label] == 0
    ]

    if absent:
        raise ValueError(
            f"Training split is missing expected classes: {absent}"
        )

    if total_rows == 0:
        raise ValueError(f"Training split is empty: {train_path}")

    X_parts = [
        reservoirs[label][:stored[label]]
        for label in class_order
    ]

    y_parts = [
        np.full(stored[label], label, dtype=object)
        for label in class_order
    ]

    X_sample = np.vstack(X_parts).astype(
        np.float32,
        copy=False,
    )
    y_sample_text = np.concatenate(y_parts).astype(str)

    sample_summary = pd.DataFrame([
        {
            "Label": label,
            "train_rows": int(train_counts[label]),
            "training_sample_rows": int(stored[label]),
            "sample_pct_of_class": (
                100.0 * stored[label] / train_counts[label]
            ),
            "max_rows_per_class": int(max_rows_per_class),
            "cap_reached": bool(
                train_counts[label] > max_rows_per_class
            ),
        }
        for label in class_order
    ])

    print(f"  Training rows scanned: {total_rows:,}")
    print(
        f"  Fixed training sample: {len(y_sample_text):,} rows "
        f"x {X_sample.shape[1]} features"
    )

    return X_sample, y_sample_text, sample_summary, total_rows


def _make_estimator(
    model_id: str,
    seed: int,
    class_count: int,
) -> BaseEstimator:
    """Return one model with fixed hyperparameters and the requested seed."""
    if model_id == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=24,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
        )

    if model_id == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )

    if model_id == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required for Phase 05. Install it in the "
                "Python (ClearFlow) environment with: "
                "python -m pip install xgboost"
            ) from exc

        cpu_count = os.cpu_count() or 1

        return XGBClassifier(
            objective="multi:softprob",
            num_class=class_count,
            eval_metric="mlogloss",
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=max(1, min(cpu_count, 8)),
            random_state=seed,
        )

    if model_id == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=500,
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                random_state=seed,
            )),
        ])

    raise ValueError(f"Unknown model ID: {model_id}")


def _atomic_joblib_dump(
    model: BaseEstimator,
    path: Path,
) -> int:
    """Save a fitted model and return its compressed file size in bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")

    joblib.dump(model, temporary_path, compress=3)
    temporary_path.replace(path)

    return path.stat().st_size


def _evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_order: list[str],
) -> tuple[
    dict[str, float],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    codes = np.arange(len(class_order), dtype=np.int32)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=codes,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=codes,
    )

    weighted_f1 = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=codes,
        average="weighted",
        zero_division=0,
    )[2]

    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(weighted_f1),
    }

    per_class_rows = [
        {
            "class_order": i,
            "Label": class_order[i],
            "test_support": int(support[i]),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1[i]),
        }
        for i in range(len(class_order))
    ]

    confusion_rows = [
        {
            "true_label": class_order[true_i],
            "predicted_label": class_order[pred_i],
            "count": int(matrix[true_i, pred_i]),
        }
        for true_i in range(len(class_order))
        for pred_i in range(len(class_order))
    ]

    return summary, per_class_rows, confusion_rows


def _save_checkpoint(
    rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
    confusion_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save results after every configuration so interrupted runs can resume."""
    _write_csv_atomic(
        pd.DataFrame(rows),
        output_dir / "experiment_summary.csv",
    )
    _write_csv_atomic(
        pd.DataFrame(per_class_rows),
        output_dir / "per_class_metrics.csv",
    )
    _write_csv_atomic(
        pd.DataFrame(confusion_rows),
        output_dir / "confusion_matrix_long.csv",
    )


def run_phase05_model_training(
    project_root: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_rows_per_class: int = DEFAULT_MAX_ROWS_PER_CLASS,
    sample_random_state: int = DEFAULT_SAMPLE_RANDOM_STATE,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    resume: bool = True,
) -> dict[str, Any]:
    """Train and evaluate all 200 configurations, saving a checkpoint per run."""
    project_root = Path(project_root).resolve()

    split_manifest_path = (
        project_root / "exports" / "phase03_split" / "split_manifest.json"
    )
    split_manifest = _read_json(split_manifest_path)

    feature_columns = list(split_manifest.get("model_features", []))
    class_order = list(split_manifest.get("class_order", []))
    target_column = split_manifest.get("target_column", TARGET_COLUMN)

    if len(feature_columns) != 70 or not class_order:
        raise ValueError(
            "Phase 03 manifest must define 70 model features and the class order."
        )

    if target_column != TARGET_COLUMN:
        raise ValueError(
            f"Expected target column {TARGET_COLUMN!r}, "
            f"got {target_column!r}."
        )

    label_to_code = {
        label: index
        for index, label in enumerate(class_order)
    }

    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError(
            "The project protocol requires five distinct random seeds."
        )

    train_path = project_root / split_manifest.get(
        "train_output", "data/processed/train.csv.gz"
    )
    test_path = project_root / split_manifest.get(
        "test_output", "data/processed/test.csv.gz"
    )

    feature_sets_path = (
        project_root / "exports" / "phase04_feature_selection" / "feature_sets.json"
    )

    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "Phase 03 train/test split files are missing. Run Phase 03 first."
        )

    feature_sets = _load_feature_sets(
        feature_sets_path,
        feature_columns,
    )

    output_dir = project_root / "exports" / "phase05_model_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = project_root / "results" / "models" / "phase05"
    model_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "experiment_summary.csv"
    per_class_path = output_dir / "per_class_metrics.csv"
    confusion_path = output_dir / "confusion_matrix_long.csv"

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    # Resume only configurations with a saved model and complete metrics.
    if (
        resume
        and summary_path.exists()
        and per_class_path.exists()
        and confusion_path.exists()
    ):
        previous_summary = pd.read_csv(summary_path)
        previous_per_class = pd.read_csv(per_class_path)
        previous_confusion = pd.read_csv(confusion_path)

        if not previous_summary.empty:
            per_class_counts = (
                previous_per_class.groupby("experiment_id")
                .size()
                .to_dict()
            )
            confusion_counts = (
                previous_confusion.groupby("experiment_id")
                .size()
                .to_dict()
            )

            for row in previous_summary.to_dict("records"):
                experiment_id = row.get("experiment_id", "")
                relative_model_path = row.get("model_artifact_path", "")

                artifact_exists = (
                    isinstance(relative_model_path, str)
                    and bool(relative_model_path)
                    and (project_root / relative_model_path).is_file()
                )

                if (
                    artifact_exists
                    and per_class_counts.get(experiment_id) == len(class_order)
                    and confusion_counts.get(experiment_id) == len(class_order) ** 2
                ):
                    summary_rows.append(row)
                    completed_ids.add(experiment_id)

            if completed_ids:
                per_class_rows = previous_per_class[
                    previous_per_class["experiment_id"].isin(completed_ids)
                ].to_dict("records")

                confusion_rows = previous_confusion[
                    previous_confusion["experiment_id"].isin(completed_ids)
                ].to_dict("records")

        print(
            "Resume enabled: "
            f"{len(completed_ids)} complete configurations already exist."
        )

    # Sample the training split once. This same sample is used for all models.
    X_train, y_train_text, sample_summary, training_rows_scanned = (
        _make_training_sample(
            train_path=train_path,
            feature_columns=feature_columns,
            target_column=target_column,
            class_order=class_order,
            max_rows_per_class=max_rows_per_class,
            chunk_size=chunk_size,
            random_state=sample_random_state,
        )
    )

    y_train = np.fromiter(
        (label_to_code[label] for label in y_train_text),
        dtype=np.int32,
        count=len(y_train_text),
    )
    del y_train_text

    sample_path = (
        project_root / "data" / "processed" / "model_training_sample.csv.gz"
    )

    sample_frame = pd.DataFrame(
        X_train,
        columns=feature_columns,
    )
    sample_frame[target_column] = [
        class_order[int(code)]
        for code in y_train
    ]
    sample_frame.to_csv(
        sample_path,
        index=False,
        compression="gzip",
    )
    del sample_frame

    _write_csv_atomic(
        sample_summary,
        output_dir / "training_sample_summary.csv",
    )

    # Class weights are calculated only from the fixed training sample.
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )
    train_rows = len(y_train)

    # Load the full test split once. It is only used for final evaluation.
    print("PASS 2/3: loading the fixed test split once for evaluation...")

    dtype_map = {
        feature: np.float32
        for feature in feature_columns
    }
    dtype_map[target_column] = "string"

    test_frame = pd.read_csv(
        test_path,
        usecols=feature_columns + [target_column],
        compression="infer",
        low_memory=False,
        dtype=dtype_map,
    )

    X_test = test_frame[feature_columns].to_numpy(
        dtype=np.float32,
        copy=False,
    )
    y_test_text = (
        test_frame[target_column]
        .astype("string")
        .str.strip()
        .to_numpy(dtype=str)
    )
    del test_frame

    if not np.isfinite(X_test).all():
        raise ValueError(
            "Non-finite model feature values found in the test split."
        )

    unknown_test_labels = sorted(
        set(y_test_text) - set(label_to_code)
    )
    if unknown_test_labels:
        raise ValueError(
            f"Unexpected labels in the test split: {unknown_test_labels}"
        )

    y_test = np.fromiter(
        (label_to_code[label] for label in y_test_text),
        dtype=np.int32,
        count=len(y_test_text),
    )
    del y_test_text

    test_rows = len(y_test)
    total_experiments = (
        len(MODEL_ORDER) * len(feature_sets) * len(seeds)
    )
    experiment_number = len(completed_ids)

    print(
        f"PASS 3/3: running {total_experiments} model configurations..."
    )

    # Loop by feature set to avoid keeping all test subsets in memory.
    feature_position = {
        name: index
        for index, name in enumerate(feature_columns)
    }

    for feature_set_id, set_info in feature_sets.items():
        selected_features = list(set_info["features"])
        selected_indices = [
            feature_position[name]
            for name in selected_features
        ]

        X_train_set = X_train[:, selected_indices]
        X_test_set = X_test[:, selected_indices]

        for model_id in MODEL_ORDER:
            for seed in seeds:
                experiment_id = (
                    f"{model_id}__{feature_set_id}__seed{seed}"
                )

                if experiment_id in completed_ids:
                    continue

                experiment_number += 1
                print(
                    f"[{experiment_number}/{total_experiments}] "
                    f"{MODEL_DISPLAY_NAMES[model_id]} | "
                    f"{feature_set_id} | seed={seed}"
                )

                estimator = _make_estimator(
                    model_id,
                    int(seed),
                    len(class_order),
                )

                start = time.perf_counter()

                if model_id == "xgboost":
                    estimator.fit(
                        X_train_set,
                        y_train,
                        sample_weight=sample_weights,
                    )
                else:
                    estimator.fit(X_train_set, y_train)

                train_seconds = time.perf_counter() - start

                start = time.perf_counter()
                predictions = np.asarray(
                    estimator.predict(X_test_set),
                    dtype=np.int32,
                )
                inference_seconds = time.perf_counter() - start

                metrics, current_per_class, current_confusion = _evaluate(
                    y_test,
                    predictions,
                    class_order,
                )

                model_path = model_dir / f"{experiment_id}.joblib"
                model_size_bytes = _atomic_joblib_dump(
                    estimator,
                    model_path,
                )
                relative_model_path = (
                    model_path.relative_to(project_root).as_posix()
                )

                common_fields = {
                    "experiment_id": experiment_id,
                    "model_id": model_id,
                    "model_name": MODEL_DISPLAY_NAMES[model_id],
                    "feature_set_id": feature_set_id,
                    "feature_set_name": set_info.get(
                        "feature_set_name",
                        feature_set_id,
                    ),
                    "selection_method": set_info.get(
                        "selection_method",
                        "unknown",
                    ),
                    "feature_count": len(selected_features),
                    "seed": int(seed),
                    "training_sample_rows": int(train_rows),
                    "test_rows": int(test_rows),
                }

                current_per_class = [
                    {**common_fields, **row}
                    for row in current_per_class
                ]
                current_confusion = [
                    {**common_fields, **row}
                    for row in current_confusion
                ]

                result_row = {
                    **common_fields,
                    **metrics,
                    "training_time_seconds": float(train_seconds),
                    "inference_time_seconds": float(inference_seconds),
                    "inference_ms_per_flow": float(
                        1000.0 * inference_seconds / test_rows
                    ),
                    "model_size_bytes": int(model_size_bytes),
                    "model_size_mb": float(
                        model_size_bytes / (1024 ** 2)
                    ),
                    "model_artifact_path": relative_model_path,
                }

                summary_rows.append(result_row)
                per_class_rows.extend(current_per_class)
                confusion_rows.extend(current_confusion)
                completed_ids.add(experiment_id)

                # Checkpoint after every completed configuration.
                _save_checkpoint(
                    summary_rows,
                    per_class_rows,
                    confusion_rows,
                    output_dir,
                )

                del estimator, predictions

        del X_train_set, X_test_set

    manifest = {
        "phase": "05_model_training_and_evaluation",
        "train_split": str(train_path.relative_to(project_root)),
        "test_split": str(test_path.relative_to(project_root)),
        "feature_sets_input": (
            "exports/phase04_feature_selection/feature_sets.json"
        ),
        "test_split_used_for_training_or_preprocessing": False,
        "test_rows_evaluated_per_configuration": int(test_rows),
        "training_rows_scanned": int(training_rows_scanned),
        "training_sample_rows": int(train_rows),
        "training_sample_path": str(sample_path.relative_to(project_root)),
        "training_sample_strategy": (
            "One fixed, uniform per-class reservoir sample drawn from the "
            "training split; up to max_rows_per_class rows per class. "
            "Reused unchanged across all configurations."
        ),
        "max_rows_per_class": int(max_rows_per_class),
        "sample_random_state": int(sample_random_state),
        "class_imbalance_handling": (
            "Training sample is class-capped; Decision Tree, Random Forest "
            "and Logistic Regression use class_weight. XGBoost uses balanced "
            "per-row sample weights. No test resampling."
        ),
        "random_seeds": [int(seed) for seed in seeds],
        "expected_configuration_count": int(total_experiments),
        "completed_configuration_count": int(len(completed_ids)),
        "models": {
            "decision_tree": {
                "max_depth": 24,
                "min_samples_leaf": 5,
                "class_weight": "balanced",
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 20,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
            },
            "xgboost": {
                "objective": "multi:softprob",
                "eval_metric": "mlogloss",
                "n_estimators": 100,
                "max_depth": 8,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "class_weighting": (
                    "compute_sample_weight(class_weight='balanced')"
                ),
            },
            "logistic_regression": {
                "scaler": "StandardScaler fitted on training sample only",
                "max_iter": 500,
                "C": 1.0,
                "class_weight": "balanced",
                "solver": "lbfgs",
            },
        },
        "timing_definition": {
            "training_time_seconds": (
                "Estimator fit() only; excludes feature-array slicing "
                "and model serialization."
            ),
            "inference_time_seconds": (
                "predict() on the full test split; excludes feature-array slicing."
            ),
            "inference_ms_per_flow": (
                "Inference time divided by total test rows, multiplied by 1000."
            ),
        },
        "model_artifacts_directory": "results/models/phase05/",
        "outputs": {
            "experiment_summary": (
                "exports/phase05_model_results/experiment_summary.csv"
            ),
            "per_class_metrics": (
                "exports/phase05_model_results/per_class_metrics.csv"
            ),
            "confusion_matrix_long": (
                "exports/phase05_model_results/confusion_matrix_long.csv"
            ),
            "training_sample_summary": (
                "exports/phase05_model_results/training_sample_summary.csv"
            ),
            "manifest": (
                "exports/phase05_model_results/model_training_manifest.json"
            ),
        },
        "caveats": [
            "Metrics are evaluated on the full, naturally imbalanced test split.",
            "Models use a fixed class-capped training sample, not every training row.",
            "Macro-F1 gives equal weight to the 12 classes; accuracy alone is insufficient.",
            "Training and inference timings are machine-specific.",
            "Inference timing excludes feature extraction and serialization.",
            "Logistic Regression may converge slowly or reach max_iter; check warnings.",
            "Random seeds should affect tree ensembles and XGBoost more than Logistic Regression.",
            "Model size is the compressed artifact size on disk, not peak process RAM.",
        ],
    }

    with (
        output_dir / "model_training_manifest.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    readme = """ClearFlow — Phase 05 Model Training and Evaluation exports

experiment_summary.csv:
One row per model x feature set x seed (200 expected).

per_class_metrics.csv:
Per-class precision, recall, F1, and test support (2,400 expected rows).

confusion_matrix_long.csv:
Long-format true/predicted class counts (28,800 expected rows).

training_sample_summary.csv:
Original versus sampled training rows per class.

model_training_manifest.json:
Split provenance, sampling strategy, seeds, parameters, timings, and caveats.

The fixed training sample contains up to 10,000 rows per class from the
training split only. The same sample is reused for all configurations.
Evaluation uses every row in the fixed test split. Logistic Regression
uses StandardScaler fitted only on the training sample.

Model artifacts are stored under results/models/phase05/ for the later
SHAP phase. model_size_mb is the compressed model file size on disk,
not peak process RAM.

Do not compare accuracy alone. Use macro-F1 and per-class recall/F1,
especially for rare attacks. Timings are hardware-dependent.
"""

    (output_dir / "README.txt").write_text(
        readme,
        encoding="utf-8",
    )

    print(
        f"PASS: {len(completed_ids)} / {total_experiments} "
        "configurations complete."
    )
    print(f"Exports: {output_dir}")
    print(f"Model artifacts: {model_dir}")

    return {
        "experiment_summary": pd.DataFrame(summary_rows),
        "per_class_metrics": pd.DataFrame(per_class_rows),
        "confusion_matrix_long": pd.DataFrame(confusion_rows),
        "training_sample_summary": sample_summary,
        "manifest": manifest,
    }
