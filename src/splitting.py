"""ClearFlow Phase 3: fixed, stratified train/test split.

The split is group-aware: identical vectors across the 70 model features are
kept on the same side of the split. This prevents repeated model inputs from
appearing in both train and test while retaining repeated observations in the
cleaned dataset. StratifiedGroupKFold with five folds targets an approximately
80/20 split while preserving class proportions as closely as group structure
allows.

Outputs:
- data/processed/train.csv.gz
- data/processed/test.csv.gz
- exports/phase03_split/split_summary_by_class.csv
- exports/phase03_split/split_summary_by_dataset.csv
- exports/phase03_split/class_distribution_by_split.csv
- exports/phase03_split/split_manifest.json

No balancing, scaling, feature selection, or model training is performed.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


EXPECTED_CLASSES = [
    "Benign",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "PortScan",
    "SSH-Patator",
    "SYN Flood",
    "Web Attack",
]

METADATA_COLUMNS = {"dataset", "source_file"}
TARGET_COLUMN = "Label"
DEFAULT_CHUNK_SIZE = 100_000
DEFAULT_RANDOM_STATE = 42
DEFAULT_FOLDS = 5


def _feature_hashes(frame: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    """Return stable row hashes for numeric model inputs only."""
    features = frame[feature_columns].astype("float64")
    return pd.util.hash_pandas_object(features, index=False).to_numpy(dtype=np.uint64)


def _discover_schema(clean_path: Path) -> tuple[list[str], list[str]]:
    header = pd.read_csv(clean_path, nrows=0, compression="gzip")
    columns = header.columns.tolist()

    if TARGET_COLUMN not in columns:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r} in {clean_path}")

    missing_metadata = METADATA_COLUMNS - set(columns)
    if missing_metadata:
        raise ValueError(f"Clean dataset is missing provenance columns: {sorted(missing_metadata)}")

    feature_columns = [
        column for column in columns
        if column not in METADATA_COLUMNS and column != TARGET_COLUMN
    ]

    if len(feature_columns) != 70:
        raise ValueError(
            f"Expected 70 retained features after Phase 2; found {len(feature_columns)}. "
            "Inspect the clean dataset and feature audit before splitting."
        )

    return columns, feature_columns


def run_phase03_split(
    project_root: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_splits: int = DEFAULT_FOLDS,
) -> dict[str, Any]:
    """Create one reproducible, group-aware stratified train/test split."""
    project_root = Path(project_root).resolve()
    clean_path = project_root / "data" / "processed" / "cleaned_flows.csv.gz"
    if not clean_path.is_file():
        raise FileNotFoundError(f"Clean Phase 2 dataset not found: {clean_path}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if n_splits != 5:
        raise ValueError("This project uses five folds to target an 80/20 split; n_splits must be 5.")

    columns, feature_columns = _discover_schema(clean_path)
    target_column = TARGET_COLUMN

    processed_dir = project_root / "data" / "processed"
    export_dir = project_root / "exports" / "phase03_split"
    processed_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train.csv.gz"
    test_path = processed_dir / "test.csv.gz"

    # Pass 1: compute a group ID from each row's model-feature vector and
    # encode its class. Only compact arrays are retained in memory.
    print("PASS 1/2: hashing model inputs and reading class labels...")
    class_to_code = {label: index for index, label in enumerate(EXPECTED_CLASSES)}
    group_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    class_counts: Counter[str] = Counter()
    n_rows = 0

    for chunk in pd.read_csv(
        clean_path,
        usecols=feature_columns + [target_column],
        chunksize=chunk_size,
        low_memory=False,
        float_precision="round_trip",
        compression="gzip",
    ):
        labels = chunk[target_column].astype("string").str.strip()
        if labels.isna().any():
            raise ValueError("Missing labels found in the Phase 2 cleaned dataset.")

        unknown = sorted(set(labels.unique()) - set(class_to_code))
        if unknown:
            raise ValueError(f"Unexpected target labels in clean dataset: {unknown}")

        codes = labels.map(class_to_code).to_numpy(dtype=np.int8)
        hashes = _feature_hashes(chunk, feature_columns)

        group_chunks.append(hashes)
        label_chunks.append(codes)
        class_counts.update(labels.value_counts().to_dict())
        n_rows += len(chunk)

    if n_rows == 0:
        raise ValueError("The cleaned dataset contains no rows.")

    groups = np.concatenate(group_chunks)
    y_codes = np.concatenate(label_chunks)
    if len(groups) != n_rows or len(y_codes) != n_rows:
        raise RuntimeError("Internal row-count mismatch while preparing split.")

    classes_in_data = [label for label in EXPECTED_CLASSES if class_counts[label] > 0]
    if len(classes_in_data) != len(EXPECTED_CLASSES):
        absent = sorted(set(EXPECTED_CLASSES) - set(classes_in_data))
        raise ValueError(f"Expected all 12 modeling classes after Phase 2; absent: {absent}")

    # Every group should have exactly one label because Phase 2 removed
    # feature vectors that conflict across labels. Validate this assumption.
    group_label_pairs = pd.DataFrame({"group": groups, "label_code": y_codes})
    labels_per_group = group_label_pairs.groupby("group", sort=False)["label_code"].nunique()
    conflicting_group_count = int((labels_per_group > 1).sum())
    if conflicting_group_count:
        raise ValueError(
            f"Found {conflicting_group_count} feature groups with conflicting labels "
            "in the cleaned data. Rerun/check Phase 2 before splitting."
        )

    groups_per_class = group_label_pairs.groupby("label_code", sort=False)["group"].nunique()
    insufficient = {
        EXPECTED_CLASSES[int(code)]: int(count)
        for code, count in groups_per_class.items()
        if count < n_splits
    }
    if insufficient:
        raise ValueError(
            "Not enough unique feature groups for a five-fold stratified split: "
            f"{insufficient}"
        )

    print(f"  Rows: {n_rows:,}")
    print(f"  Model features: {len(feature_columns)}")
    print(f"  Unique feature groups: {len(np.unique(groups)):,}")

    # Evaluate all five candidate folds and select the one with
    # the best per-class balance while keeping feature groups intact.
    print("PASS 2/2: comparing candidate folds...")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    dummy_x = np.zeros(n_rows, dtype=np.uint8)

    overall_class_counts = np.bincount(
        y_codes,
        minlength=len(EXPECTED_CLASSES),
    )

    target_fraction = 1.0 / n_splits
    best_score = None
    train_indices = None
    test_indices = None
    fold_results = []

    for fold_number, (candidate_train, candidate_test) in enumerate(
        splitter.split(dummy_x, y_codes, groups=groups),
        start=1,
    ):
        test_class_counts = np.bincount(
            y_codes[candidate_test],
            minlength=len(EXPECTED_CLASSES),
        )

        class_test_fractions = (
            test_class_counts / overall_class_counts
        )

        class_errors = np.abs(
            class_test_fractions - target_fraction
        )

        total_size_error = abs(
            len(candidate_test) / n_rows - target_fraction
        )

        # Prioritize the largest class deviation, then mean deviation,
        # then overall test-size deviation.
        score = (
            float(class_errors.max()),
            float(class_errors.mean()),
            float(total_size_error),
        )

        fold_results.append({
            "fold": fold_number,
            "test_rows": len(candidate_test),
            "test_pct": 100 * len(candidate_test) / n_rows,
            "max_class_deviation_pp": 100 * class_errors.max(),
            "mean_class_deviation_pp": 100 * class_errors.mean(),
        })

        if best_score is None or score < best_score:
            best_score = score
            train_indices = candidate_train
            test_indices = candidate_test
            selected_fold = fold_number

    fold_comparison = pd.DataFrame(fold_results)

    print("Candidate fold comparison:")
    print(fold_comparison.to_string(index=False))
    print(f"\nSelected fold: {selected_fold}")

    is_test = np.zeros(n_rows, dtype=bool)
    is_test[test_indices] = True

    # Explicitly verify no feature group crosses the split.
    train_groups = np.unique(groups[train_indices])
    test_groups = np.unique(groups[test_indices])
    group_overlap = np.intersect1d(train_groups, test_groups, assume_unique=True)
    if len(group_overlap):
        raise RuntimeError(f"Feature-group leakage detected: {len(group_overlap)} groups overlap.")

    # Rewrite the cleaned dataset in source order, streaming rows to gzip.
    train_path.unlink(missing_ok=True)
    test_path.unlink(missing_ok=True)
    split_class_counts: Counter[tuple[str, str, str]] = Counter()
    split_dataset_counts: Counter[tuple[str, str]] = Counter()
    split_source_counts: Counter[tuple[str, str, str]] = Counter()
    offset = 0
    train_header_written = False
    test_header_written = False

    with gzip.open(train_path, "wt", encoding="utf-8", newline="") as train_output, \
         gzip.open(test_path, "wt", encoding="utf-8", newline="") as test_output:
        for chunk in pd.read_csv(
            clean_path,
            chunksize=chunk_size,
            low_memory=False,
            float_precision="round_trip",
            compression="gzip",
        ):
            end = offset + len(chunk)
            test_mask_chunk = is_test[offset:end]
            train_mask_chunk = ~test_mask_chunk

            train_chunk = chunk.loc[train_mask_chunk, columns]
            test_chunk = chunk.loc[test_mask_chunk, columns]

            if not train_chunk.empty:
                train_chunk.to_csv(
                    train_output,
                    index=False,
                    header=not train_header_written,
                    float_format="%.17g",
                )
                train_header_written = True

            if not test_chunk.empty:
                test_chunk.to_csv(
                    test_output,
                    index=False,
                    header=not test_header_written,
                    float_format="%.17g",
                )
                test_header_written = True

            for split_name, part in (("train", train_chunk), ("test", test_chunk)):
                if part.empty:
                    continue
                for (dataset, label), count in part.groupby(["dataset", target_column], sort=False).size().items():
                    split_class_counts[(split_name, str(dataset), str(label))] += int(count)
                for dataset, count in part["dataset"].value_counts().items():
                    split_dataset_counts[(split_name, str(dataset))] += int(count)
                for (dataset, source_file), count in part.groupby(["dataset", "source_file"], sort=False).size().items():
                    split_source_counts[(split_name, str(dataset), str(source_file))] += int(count)

            offset = end

    if offset != n_rows:
        raise RuntimeError(f"Split writer read {offset:,} rows, expected {n_rows:,}.")

    train_rows = int(len(train_indices))
    test_rows = int(len(test_indices))
    observed_classes = pd.DataFrame(
        [
            {"split": split, "dataset": dataset, "Label": label, "rows": count}
            for (split, dataset, label), count in split_class_counts.items()
        ]
    )
    if observed_classes.empty:
        raise RuntimeError("No class counts were produced for the split.")

    # Class distributions for audit/reproducibility, not a visualization requirement.
    split_totals = observed_classes.groupby("split")["rows"].sum().to_dict()
    dataset_totals = observed_classes.groupby(["split", "dataset"])["rows"].sum()
    observed_classes["pct_within_split"] = observed_classes.apply(
        lambda row: 100.0 * row["rows"] / split_totals[row["split"]], axis=1
    )
    observed_classes["pct_within_dataset_split"] = observed_classes.apply(
        lambda row: 100.0 * row["rows"] / dataset_totals[(row["split"], row["dataset"])],
        axis=1,
    )
    observed_classes["pct_of_class_in_split"] = observed_classes.apply(
        lambda row: 100.0 * row["rows"] / class_counts[row["Label"]], axis=1
    )
    observed_classes = observed_classes.sort_values(["split", "dataset", "Label"])
    observed_classes.to_csv(export_dir / "class_distribution_by_split.csv", index=False)

    summary_class_rows: list[dict[str, Any]] = []
    for label in EXPECTED_CLASSES:
        train_count = sum(
            count for (split, _, current_label), count in split_class_counts.items()
            if split == "train" and current_label == label
        )
        test_count = sum(
            count for (split, _, current_label), count in split_class_counts.items()
            if split == "test" and current_label == label
        )
        total = train_count + test_count
        summary_class_rows.append({
            "Label": label,
            "train_rows": train_count,
            "test_rows": test_count,
            "total_rows": total,
            "test_pct_of_class": 100.0 * test_count / total if total else 0.0,
            "class_pct_of_clean_dataset": 100.0 * total / n_rows,
        })
    summary_by_class = pd.DataFrame(summary_class_rows)
    summary_by_class.to_csv(export_dir / "split_summary_by_class.csv", index=False)

    summary_dataset_rows: list[dict[str, Any]] = []
    for dataset in sorted({key[1] for key in split_dataset_counts}):
        train_count = split_dataset_counts.get(("train", dataset), 0)
        test_count = split_dataset_counts.get(("test", dataset), 0)
        total = train_count + test_count
        summary_dataset_rows.append({
            "dataset": dataset,
            "train_rows": train_count,
            "test_rows": test_count,
            "total_rows": total,
            "train_pct": 100.0 * train_count / total if total else 0.0,
            "test_pct": 100.0 * test_count / total if total else 0.0,
        })
    summary_by_dataset = pd.DataFrame(summary_dataset_rows)
    summary_by_dataset.to_csv(export_dir / "split_summary_by_dataset.csv", index=False)

    source_summary_rows = []
    for (split, dataset, source_file), count in sorted(split_source_counts.items()):
        source_summary_rows.append({
            "split": split,
            "dataset": dataset,
            "source_file": source_file,
            "rows": count,
        })
    pd.DataFrame(source_summary_rows).to_csv(
        export_dir / "split_summary_by_source_file.csv", index=False
    )

    manifest = {
        "phase": "03_train_test_split",
        "input_clean_dataset": str(clean_path.relative_to(project_root)),
        "train_output": str(train_path.relative_to(project_root)),
        "test_output": str(test_path.relative_to(project_root)),
        "rows_total": n_rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_pct": 100.0 * train_rows / n_rows,
        "test_pct": 100.0 * test_rows / n_rows,
        "intended_split": "80/20",
        "split_method": (
            "StratifiedGroupKFold; best-balanced of five candidate folds"
        ),
        "selected_fold": selected_fold,
        "n_splits": n_splits,
        "random_state": random_state,
        "grouping_rule": "Rows with identical 70-feature model input vectors are kept in the same split.",
        "unique_feature_groups": int(len(np.unique(groups))),
        "train_unique_feature_groups": int(len(train_groups)),
        "test_unique_feature_groups": int(len(test_groups)),
        "feature_group_overlap": int(len(group_overlap)),
        "model_feature_count": len(feature_columns),
        "model_features": feature_columns,
        "target_column": target_column,
        "provenance_columns_not_for_modeling": sorted(METADATA_COLUMNS),
        "class_order": EXPECTED_CLASSES,
        "class_counts": {label: int(class_counts[label]) for label in EXPECTED_CLASSES},
        "balancing_applied": False,
        "scaling_applied": False,
        "feature_selection_applied": False,
        "model_training_applied": False,
    }
    with (export_dir / "split_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "clean_path": clean_path,
        "train_path": train_path,
        "test_path": test_path,
        "export_dir": export_dir,
        "summary_by_class": summary_by_class,
        "summary_by_dataset": summary_by_dataset,
        "class_distribution": observed_classes,
        "manifest": manifest,
    }
