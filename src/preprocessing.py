"""ClearFlow Phase 2: reproducible, chunked dataset preprocessing.

This module:
1. Harmonizes CICIDS-2017 and SDN TCP-SYN schemas and labels.
2. Removes rows with missing labels, excluded classes, or invalid numeric features.
3. Deduplicates exact raw records within each source dataset.
4. Removes feature vectors whose labels conflict after metadata removal.
5. Removes zero-variance features from the remaining dataset.
6. Exports a compressed clean dataset and BI-friendly summary/sample tables.

Destination Port is deliberately retained. Dataset and source_file columns in
output are provenance metadata, not model features. No class balancing, train/test
split, or scaling is performed in this phase.
"""

from __future__ import annotations

import gzip
import os
import sqlite3
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from src.schema import SDN_TO_CICIDS
except ImportError as exc:  # pragma: no cover - environment-specific guidance
    raise ImportError(
        "Cannot import SDN_TO_CICIDS from src.schema. Ensure the repository root "
        "is on sys.path and src/schema.py exists."
    ) from exc


EXCLUDED_CLASSES = {"Heartbleed", "Infiltration"}
CHUNK_SIZE_DEFAULT = 100_000
SAMPLE_PER_CHUNK_CLASS_DEFAULT = 200
RANDOM_SEED_DEFAULT = 42

# Identifier/metadata fields removed from model features. Destination Port is
# intentionally NOT excluded, based on the project's schema/conflict audit.
EXCLUDED_SOURCE_COLUMNS = {
    "label",
    "flow id",
    "src ip",
    "dst ip",
    "source ip",
    "destination ip",
    "src port",
    "source port",
    "protocol",
    "timestamp",
    "fwd header length.1",
}

LABEL_MAP = {
    "benign": "Benign",
    "ddos": "DDoS",
    "portscan": "PortScan",
    "dos hulk": "DoS Hulk",
    "dos goldeneye": "DoS GoldenEye",
    "dos slowloris": "DoS slowloris",
    "dos slowhttptest": "DoS Slowhttptest",
    "ftp-patator": "FTP-Patator",
    "ssh-patator": "SSH-Patator",
    "bot": "Bot",
    "heartbleed": "Heartbleed",
    "infiltration": "Infiltration",
    "web attack - brute force": "Web Attack",
    "web attack - sql injection": "Web Attack",
    "web attack - xss": "Web Attack",
}


def _norm(value: Any) -> str:
    return str(value).strip().casefold()


def _map_label(value: Any, dataset: str) -> str | None:
    if pd.isna(value):
        return None

    label = str(value).strip().replace("\ufffd", "-")
    if label.casefold() in {"", "<missing>"}:
        return None

    key = label.casefold()

    if dataset == "SDN TCP-SYN":
        if key == "normal":
            return "Benign"
        if key == "ddos":
            return "SYN Flood"

    try:
        return LABEL_MAP[key]
    except KeyError as exc:
        raise ValueError(
            f"Unrecognized raw label {value!r} in dataset {dataset!r}. "
            "Do not silently discard an unseen class; update LABEL_MAP first."
        ) from exc


def _discover_inputs(project_root: Path) -> dict[str, list[Path]]:
    dirs = {
        "CICIDS-2017": project_root / "Datasets" / "CICIDS2017",
        "SDN TCP-SYN": project_root / "Datasets" / "SDN_TCP_SYN",
    }
    found: dict[str, list[Path]] = {}
    for dataset, directory in dirs.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")
        paths = sorted(directory.glob("*.csv"))
        if not paths:
            raise FileNotFoundError(f"No CSV files found in {directory}")
        found[dataset] = paths
    return found


def _build_feature_schema(
    input_files: dict[str, list[Path]],
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Return canonical features, normalized canonical names, and SDN aliases."""
    reference = input_files["CICIDS-2017"][0]
    cicids_columns = pd.read_csv(
        reference, nrows=0, encoding="utf-8-sig"
    ).columns.tolist()

    canonical_lookup: dict[str, str] = {}
    for column in cicids_columns:
        name = _norm(column)
        if name in EXCLUDED_SOURCE_COLUMNS:
            continue
        canonical_lookup[name] = str(column).strip()

    feature_columns = list(canonical_lookup.values())
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Duplicate canonical feature names after normalization.")

    aliases = {
        _norm(source): _norm(target)
        for source, target in SDN_TO_CICIDS.items()
    }

    # The current project schema should contain 77 features: 76 previously
    # aligned numerical features plus the retained Destination Port.
    if len(feature_columns) != 77:
        raise ValueError(
            f"Expected 77 candidate features after exclusions; found "
            f"{len(feature_columns)}. Recheck the CSV schema and column policy."
        )

    # Validate headers before processing millions of rows.
    expected = set(canonical_lookup)
    for dataset, paths in input_files.items():
        for path in paths:
            columns = pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns.tolist()
            labels = [c for c in columns if _norm(c) == "label"]
            if len(labels) != 1:
                raise ValueError(f"Expected one Label column in {path.name}; found {labels}")

            mapped: dict[str, str] = {}
            for column in columns:
                source = _norm(column)
                if source in EXCLUDED_SOURCE_COLUMNS:
                    continue
                target = aliases.get(source, source) if dataset == "SDN TCP-SYN" else source
                if target not in canonical_lookup:
                    raise ValueError(
                        f"Unmapped feature {column!r} in {path.name}. "
                        "Update the explicit schema mapping."
                    )
                mapped[column] = canonical_lookup[target]

            if len(set(mapped.values())) != len(mapped):
                raise ValueError(f"Feature-name collision in {path.name}")
            if set(mapped.values()) != set(feature_columns):
                raise ValueError(
                    f"Schema mismatch in {path.name}; missing="
                    f"{sorted(set(feature_columns) - set(mapped.values()))}, extra="
                    f"{sorted(set(mapped.values()) - set(feature_columns))}"
                )

    return feature_columns, canonical_lookup, aliases


def _harmonize_chunk(
    frame: pd.DataFrame,
    dataset: str,
    feature_columns: list[str],
    canonical_lookup: dict[str, str],
    aliases: dict[str, str],
) -> tuple[pd.DataFrame, pd.Series]:
    label_cols = [c for c in frame.columns if _norm(c) == "label"]
    if len(label_cols) != 1:
        raise ValueError(f"Expected exactly one Label column; found {label_cols}")
    label_col = label_cols[0]
    labels = frame[label_col].map(lambda value: _map_label(value, dataset))

    rename: dict[str, str] = {}
    for column in frame.columns:
        source = _norm(column)
        if source in EXCLUDED_SOURCE_COLUMNS:
            continue
        target = aliases.get(source, source) if dataset == "SDN TCP-SYN" else source
        if target not in canonical_lookup:
            raise ValueError(f"Unmapped feature {column!r} in {dataset}")
        rename[column] = canonical_lookup[target]

    if len(set(rename.values())) != len(rename):
        raise ValueError(f"Feature-name collision in {dataset}")
    X = frame[list(rename)].rename(columns=rename)
    X = X[feature_columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).astype("float64")
    return X, labels


def _existing_hashes(
    connection: sqlite3.Connection,
    dataset: str,
    hashes: list[str],
) -> set[str]:
    """Query SQLite in batches below SQLite's parameter limit."""
    existing: set[str] = set()
    batch_size = 800
    for start in range(0, len(hashes), batch_size):
        batch = hashes[start : start + batch_size]
        placeholders = ",".join("?" for _ in batch)
        query = (
            "SELECT row_hash FROM seen_raw_rows "
            f"WHERE dataset = ? AND row_hash IN ({placeholders})"
        )
        rows = connection.execute(query, [dataset, *batch]).fetchall()
        existing.update(row[0] for row in rows)
    return existing


def _feature_hashes(frame: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    # The CSV outputs use 17 significant digits and round-trip parsing to keep
    # these signatures stable across the conflict-detection passes.
    features = frame[feature_columns].astype("float64")
    return pd.util.hash_pandas_object(features, index=False).astype("uint64").astype(str)


def run_preprocessing(
    project_root: str | Path,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    sample_per_chunk_class: int = SAMPLE_PER_CHUNK_CLASS_DEFAULT,
    random_seed: int = RANDOM_SEED_DEFAULT,
) -> dict[str, Any]:
    """Run the full Phase 2 cleaning pipeline and return summary DataFrames."""
    project_root = Path(project_root).resolve()
    if chunk_size <= 0 or sample_per_chunk_class <= 0:
        raise ValueError("chunk_size and sample_per_chunk_class must be positive")

    input_files = _discover_inputs(project_root)
    feature_columns, canonical_lookup, aliases = _build_feature_schema(input_files)

    interim_dir = project_root / "data" / "interim"
    processed_dir = project_root / "data" / "processed"
    export_dir = project_root / "exports" / "phase02_clean"
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    candidate_path = interim_dir / "phase02_deduplicated.csv.gz"
    conflict_free_path = interim_dir / "phase02_conflict_free.csv.gz"
    clean_path = processed_dir / "cleaned_flows.csv.gz"
    for path in (candidate_path, conflict_free_path, clean_path):
        path.unlink(missing_ok=True)

    file_stats: dict[tuple[str, str], dict[str, int]] = {}
    raw_class_counts: Counter[tuple[str, str, str]] = Counter()

    with tempfile.TemporaryDirectory(prefix="clearflow_phase02_") as temp_dir:
        db_path = Path(temp_dir) / "preprocessing.sqlite"
        connection = sqlite3.connect(db_path)
        try:
            connection.executescript("""
                CREATE TABLE seen_raw_rows (
                    dataset TEXT NOT NULL,
                    row_hash TEXT NOT NULL,
                    PRIMARY KEY (dataset, row_hash)
                ) WITHOUT ROWID;

                CREATE TABLE feature_label_counts (
                    feature_hash TEXT NOT NULL,
                    label TEXT NOT NULL,
                    occurrences INTEGER NOT NULL,
                    PRIMARY KEY (feature_hash, label)
                ) WITHOUT ROWID;
            """)

            # ---------------------------------------------------------
            # PASS 1: validate rows and deduplicate exact original rows.
            # ---------------------------------------------------------
            print("PASS 1/4: cleaning invalid rows and removing exact raw duplicates...")
            first_write = True
            with gzip.open(candidate_path, "wt", encoding="utf-8", newline="") as output:
                for dataset, paths in input_files.items():
                    for path in paths:
                        key = (dataset, path.name)
                        stats = file_stats.setdefault(key, {
                            "raw_rows": 0,
                            "missing_label_rows": 0,
                            "excluded_class_rows": 0,
                            "invalid_feature_rows": 0,
                            "eligible_before_dedup": 0,
                            "exact_duplicate_rows_removed": 0,
                            "rows_after_exact_dedup": 0,
                            "conflicting_feature_rows_removed": 0,
                            "rows_after_conflict_removal": 0,
                            "final_rows": 0,
                        })
                        print(f"  {path.name}")

                        for chunk in pd.read_csv(
                            path,
                            chunksize=chunk_size,
                            dtype=str,
                            keep_default_na=False,
                            low_memory=False,
                            encoding="utf-8-sig",
                        ):
                            stats["raw_rows"] += len(chunk)
                            X, y = _harmonize_chunk(
                                chunk, dataset, feature_columns,
                                canonical_lookup, aliases,
                            )

                            nonmissing_label = y.notna()
                            stats["missing_label_rows"] += int((~nonmissing_label).sum())
                            excluded = y.isin(EXCLUDED_CLASSES).fillna(False)
                            stats["excluded_class_rows"] += int(excluded.sum())

                            for label, count in y[nonmissing_label].value_counts().items():
                                raw_class_counts[(dataset, path.name, str(label))] += int(count)

                            eligible_by_label = nonmissing_label & ~excluded
                            invalid_features = X.isna().any(axis=1)
                            stats["invalid_feature_rows"] += int(
                                (eligible_by_label & invalid_features).sum()
                            )
                            eligible = eligible_by_label & ~invalid_features
                            eligible_count = int(eligible.sum())
                            stats["eligible_before_dedup"] += eligible_count
                            if eligible_count == 0:
                                continue

                            raw_eligible = chunk.loc[eligible]
                            hashes = pd.util.hash_pandas_object(
                                raw_eligible, index=False
                            ).astype("uint64").astype(str)
                            local_first = ~hashes.duplicated(keep="first")
                            unique_hashes = hashes.loc[local_first].tolist()
                            existing = _existing_hashes(
                                connection, dataset, unique_hashes
                            )
                            new_hashes = [h for h in unique_hashes if h not in existing]
                            new_hash_set = set(new_hashes)

                            if new_hashes:
                                connection.executemany(
                                    "INSERT OR IGNORE INTO seen_raw_rows(dataset, row_hash) VALUES (?, ?)",
                                    [(dataset, value) for value in new_hashes],
                                )
                                connection.commit()

                            keep = local_first & hashes.isin(new_hash_set)
                            keep_index = hashes.index[keep]
                            duplicate_count = eligible_count - len(keep_index)
                            stats["exact_duplicate_rows_removed"] += duplicate_count
                            stats["rows_after_exact_dedup"] += len(keep_index)

                            if len(keep_index) == 0:
                                continue

                            out = X.loc[keep_index].copy()
                            out.insert(0, "source_file", path.name)
                            out.insert(0, "dataset", dataset)
                            out["Label"] = y.loc[keep_index].astype(str)
                            out = out[["dataset", "source_file", *feature_columns, "Label"]]
                            out.to_csv(
                                output, index=False, header=first_write,
                                float_format="%.17g",
                            )
                            first_write = False

            if first_write:
                raise RuntimeError("No valid rows survived the first preprocessing pass.")

            # ---------------------------------------------------------
            # PASS 2: count labels per harmonized feature vector.
            # ---------------------------------------------------------
            print("PASS 2/4: detecting conflicting labels for identical model inputs...")
            for chunk in pd.read_csv(
                candidate_path,
                chunksize=chunk_size,
                low_memory=False,
                float_precision="round_trip",
                compression="gzip",
            ):
                hashes = _feature_hashes(chunk, feature_columns)
                pairs = pd.DataFrame({
                    "feature_hash": hashes.to_numpy(),
                    "label": chunk["Label"].astype(str).to_numpy(),
                })
                grouped = (
                    pairs.groupby(["feature_hash", "label"], sort=False)
                    .size().reset_index(name="occurrences")
                )
                connection.executemany(
                    """INSERT INTO feature_label_counts(feature_hash, label, occurrences)
                       VALUES (?, ?, ?)
                       ON CONFLICT(feature_hash, label) DO UPDATE SET
                       occurrences = occurrences + excluded.occurrences""",
                    [
                        (r.feature_hash, r.label, int(r.occurrences))
                        for r in grouped.itertuples(index=False)
                    ],
                )
                connection.commit()

            conflict_hash_rows = connection.execute("""
                SELECT feature_hash
                FROM feature_label_counts
                GROUP BY feature_hash
                HAVING COUNT(DISTINCT label) > 1
            """).fetchall()
            conflict_hashes = {row[0] for row in conflict_hash_rows}

            conflict_pairs = pd.read_sql_query("""
                SELECT
                    a.label AS label_a,
                    b.label AS label_b,
                    COUNT(*) AS conflicting_patterns,
                    SUM(a.occurrences) AS rows_label_a,
                    SUM(b.occurrences) AS rows_label_b
                FROM feature_label_counts a
                JOIN feature_label_counts b
                  ON a.feature_hash = b.feature_hash
                 AND a.label < b.label
                GROUP BY a.label, b.label
                ORDER BY conflicting_patterns DESC
            """, connection)
            conflict_pairs.to_csv(
                export_dir / "ambiguous_label_pairs.csv", index=False
            )
            conflict_affected_rows = int(connection.execute("""
                WITH patterns AS (
                    SELECT feature_hash, SUM(occurrences) AS n
                    FROM feature_label_counts
                    GROUP BY feature_hash
                    HAVING COUNT(DISTINCT label) > 1
                )
                SELECT COALESCE(SUM(n), 0) FROM patterns
            """).fetchone()[0])
            pd.DataFrame([{
                "conflicting_feature_patterns": len(conflict_hashes),
                "ambiguous_rows_removed_after_exact_dedup": conflict_affected_rows,
            }]).to_csv(
                export_dir / "ambiguous_feature_summary.csv", index=False
            )
            print(
                f"  Conflicting feature patterns: {len(conflict_hashes):,}; "
                f"rows to exclude: {conflict_affected_rows:,}"
            )

            # ---------------------------------------------------------
            # PASS 3: remove ambiguous feature vectors and find constants.
            # ---------------------------------------------------------
            print("PASS 3/4: excluding ambiguous feature vectors and profiling retained features...")
            min_values = pd.Series(np.inf, index=feature_columns, dtype="float64")
            max_values = pd.Series(-np.inf, index=feature_columns, dtype="float64")
            first_write = True

            with gzip.open(conflict_free_path, "wt", encoding="utf-8", newline="") as output:
                for chunk in pd.read_csv(
                    candidate_path,
                    chunksize=chunk_size,
                    low_memory=False,
                    float_precision="round_trip",
                    compression="gzip",
                ):
                    hashes = _feature_hashes(chunk, feature_columns)
                    ambiguous = hashes.isin(conflict_hashes)
                    if ambiguous.any():
                        removed = chunk.loc[ambiguous]
                        for (dataset, source_file), count in removed.groupby(
                            ["dataset", "source_file"], sort=False
                        ).size().items():
                            file_stats[(dataset, source_file)]["conflicting_feature_rows_removed"] += int(count)

                    kept = chunk.loc[~ambiguous].copy()
                    if kept.empty:
                        continue

                    features = kept[feature_columns].astype("float64")
                    min_values = np.minimum(min_values, features.min(axis=0))
                    max_values = np.maximum(max_values, features.max(axis=0))

                    for (dataset, source_file), count in kept.groupby(
                        ["dataset", "source_file"], sort=False
                    ).size().items():
                        file_stats[(dataset, source_file)]["rows_after_conflict_removal"] += int(count)

                    kept.to_csv(
                        output, index=False, header=first_write,
                        float_format="%.17g",
                    )
                    first_write = False

            if first_write:
                raise RuntimeError("No rows remain after ambiguous-label filtering.")

            constant_features = [
                feature for feature in feature_columns
                if np.isfinite(min_values[feature])
                and min_values[feature] == max_values[feature]
            ]
            retained_features = [
                feature for feature in feature_columns
                if feature not in constant_features
            ]

            feature_audit = pd.DataFrame({
                "feature": feature_columns,
                "min_after_cleaning": [min_values[c] for c in feature_columns],
                "max_after_cleaning": [max_values[c] for c in feature_columns],
                "zero_variance": [c in constant_features for c in feature_columns],
                "decision": [
                    "removed: zero variance" if c in constant_features else "retained"
                    for c in feature_columns
                ],
                "note": [
                    "Destination Port retained after the schema/conflict sensitivity audit."
                    if _norm(c) == "destination port" else ""
                    for c in feature_columns
                ],
            })
            feature_audit.to_csv(
                export_dir / "feature_preprocessing_audit.csv", index=False
            )

            # Document column exclusions/deviations explicitly.
            column_policy_rows = [
                {
                    "column": "Destination Port",
                    "decision": "retained",
                    "reason": "Retained after sensitivity audit; excluding it increased ambiguous feature vectors substantially.",
                },
                {
                    "column": "Fwd Header Length.1",
                    "decision": "excluded",
                    "reason": "Exact duplicate of Fwd Header Length in all eight CICIDS-2017 files.",
                },
                {
                    "column": "Flow ID, Src IP, Dst IP, Src Port, Protocol, Timestamp",
                    "decision": "excluded",
                    "reason": "Identifiers or metadata, not retained as model features.",
                },
            ]
            pd.DataFrame(column_policy_rows).to_csv(
                export_dir / "column_policy.csv", index=False
            )
            print(f"  Zero-variance features removed: {len(constant_features)}")
            if constant_features:
                print("  ", ", ".join(constant_features))

            # ---------------------------------------------------------
            # PASS 4: create final clean data, BI sample, profiles, counts.
            # ---------------------------------------------------------
            print("PASS 4/4: writing final clean dataset and BI exports...")
            clean_class_counts: Counter[tuple[str, str, str]] = Counter()
            clean_file_counts: Counter[tuple[str, str]] = Counter()
            sample_parts: list[pd.DataFrame] = []
            profile_stats: dict[str, dict[str, Any]] = {}
            first_write = True

            with gzip.open(clean_path, "wt", encoding="utf-8", newline="") as output:
                for chunk_number, chunk in enumerate(pd.read_csv(
                    conflict_free_path,
                    chunksize=chunk_size,
                    low_memory=False,
                    float_precision="round_trip",
                    compression="gzip",
                )):
                    clean = chunk.drop(columns=constant_features).copy()
                    clean_columns = [
                        "dataset", "source_file", *retained_features, "Label"
                    ]
                    clean = clean[clean_columns]
                    clean.to_csv(
                        output, index=False, header=first_write,
                        float_format="%.17g",
                    )
                    first_write = False

                    for (dataset, label), count in clean.groupby(
                        ["dataset", "Label"], sort=False
                    ).size().items():
                        # Accumulate clean class counts by dataset across files.
                        clean_class_counts[(dataset, "__ALL_FILES__", str(label))] += int(count)

                    for (dataset, source_file, label), count in clean.groupby(
                        ["dataset", "source_file", "Label"], sort=False
                    ).size().items():
                        clean_class_counts[(dataset, source_file, str(label))] += int(count)
                    for (dataset, source_file), count in clean.groupby(
                        ["dataset", "source_file"], sort=False
                    ).size().items():
                        clean_file_counts[(dataset, source_file)] += int(count)
                        file_stats[(dataset, source_file)]["final_rows"] += int(count)

                    # Full-clean-data statistics by dataset/feature, calculated in chunks.
                    for dataset, subset in clean.groupby("dataset", sort=False):
                        values = subset[retained_features].astype("float64")
                        if dataset not in profile_stats:
                            profile_stats[dataset] = {
                                "count": pd.Series(0, index=retained_features, dtype="int64"),
                                "sum": pd.Series(0.0, index=retained_features, dtype="float64"),
                                "min": pd.Series(np.inf, index=retained_features, dtype="float64"),
                                "max": pd.Series(-np.inf, index=retained_features, dtype="float64"),
                                "zeros": pd.Series(0, index=retained_features, dtype="int64"),
                            }
                        profile = profile_stats[dataset]
                        profile["count"] += len(values)
                        profile["sum"] += values.sum(axis=0)
                        profile["min"] = np.minimum(profile["min"], values.min(axis=0))
                        profile["max"] = np.maximum(profile["max"], values.max(axis=0))
                        profile["zeros"] += (values == 0).sum(axis=0).astype("int64")

                    # BI sample: cap the sample per class per chunk so small classes
                    # stay visible. Do not use this sample to compute class counts.
                    for (dataset, label), group in clean.groupby(
                        ["dataset", "Label"], sort=False
                    ):
                        n = min(sample_per_chunk_class, len(group))
                        sample_parts.append(group.sample(
                            n=n,
                            random_state=random_seed + chunk_number,
                        ))

            if first_write:
                raise RuntimeError("Final clean dataset is empty.")

            # Summaries for Tableau/Power BI.
            summary_by_file = pd.DataFrame([
                {"dataset": dataset, "file": filename, **stats}
                for (dataset, filename), stats in sorted(file_stats.items())
            ])
            summary_by_file.to_csv(
                export_dir / "preprocessing_summary_by_file.csv", index=False
            )

            # Totals are sums of mutually exclusive stage removals.
            removal_columns = [
                "raw_rows", "missing_label_rows", "excluded_class_rows",
                "invalid_feature_rows", "eligible_before_dedup",
                "exact_duplicate_rows_removed", "rows_after_exact_dedup",
                "conflicting_feature_rows_removed", "rows_after_conflict_removal",
                "final_rows",
            ]
            totals = (
                summary_by_file.groupby("dataset", as_index=False)[removal_columns]
                .sum()
            )
            totals.loc[len(totals)] = {
                "dataset": "ALL",
                **{column: int(summary_by_file[column].sum()) for column in removal_columns},
            }
            totals.to_csv(export_dir / "preprocessing_summary_totals.csv", index=False)

            # Class counts for raw harmonized labels vs cleaned data.
            class_rows = []
            for (dataset, filename, label), count in raw_class_counts.items():
                denominator = file_stats[(dataset, filename)]["raw_rows"]
                class_rows.append({
                    "stage": "Raw (harmonized labels)",
                    "dataset": dataset,
                    "source_file": filename,
                    "class": label,
                    "flow_count": int(count),
                    "file_rows": int(denominator),
                    "percentage_of_file": 100 * count / denominator if denominator else 0,
                })
            for (dataset, filename, label), count in clean_class_counts.items():
                if filename == "__ALL_FILES__":
                    continue
                denominator = clean_file_counts[(dataset, filename)]
                class_rows.append({
                    "stage": "Cleaned",
                    "dataset": dataset,
                    "source_file": filename,
                    "class": label,
                    "flow_count": int(count),
                    "file_rows": int(denominator),
                    "percentage_of_file": 100 * count / denominator if denominator else 0,
                })
            class_stage = pd.DataFrame(class_rows)
            class_stage.to_csv(export_dir / "class_counts_before_after.csv", index=False)

            # Global class counts by dataset, suitable for a clean class chart.
            clean_counts_dataset = pd.DataFrame([
                {"dataset": ds, "class": label, "flow_count": int(count)}
                for (ds, filename, label), count in clean_class_counts.items()
                if filename == "__ALL_FILES__"
            ])
            clean_counts_dataset.to_csv(
                export_dir / "clean_class_counts_by_dataset.csv", index=False
            )

            # Full clean feature profile by dataset.
            profile_rows = []
            for dataset, profile in profile_stats.items():
                for feature in retained_features:
                    count = int(profile["count"][feature])
                    zero_count = int(profile["zeros"][feature])
                    profile_rows.append({
                        "dataset": dataset,
                        "feature": feature,
                        "rows": count,
                        "min": float(profile["min"][feature]),
                        "max": float(profile["max"][feature]),
                        "mean": float(profile["sum"][feature] / count) if count else np.nan,
                        "zero_count": zero_count,
                        "zero_pct": 100 * zero_count / count if count else np.nan,
                    })
            pd.DataFrame(profile_rows).to_csv(
                export_dir / "clean_feature_profile_by_dataset.csv", index=False
            )

            # BI visualization sample, with full retained feature set.
            sample = pd.concat(sample_parts, ignore_index=True)
            sample.insert(2, "sample_id", [f"clean_{i:07d}" for i in range(len(sample))])
            sample = sample.rename(columns={"Label": "label"})
            sample["display_class"] = sample["label"]
            sample.to_csv(
                export_dir / "clean_visualization_sample_wide.csv",
                index=False,
                encoding="utf-8",
                float_format="%.17g",
            )

            # Match raw-phase feature names for direct before/after comparisons.
            selected_specs = {
                "flow_duration": "Flow Duration",
                "forward_packet_count": "Total Fwd Packets",
                "flow_byte_rate": "Flow Bytes/s",
            }
            absent_selected = [c for c in selected_specs.values() if c not in retained_features]
            if absent_selected:
                raise ValueError(
                    f"Cannot make comparable distribution export; missing: {absent_selected}"
                )

            feature_sample = sample[[
                "dataset", "sample_id", "source_file", "label", "display_class",
                *selected_specs.values(),
            ]].copy()
            feature_sample = feature_sample.rename(
                columns={source: target for target, source in selected_specs.items()}
            )
            feature_sample.to_csv(
                export_dir / "clean_feature_distribution_sample_wide.csv",
                index=False,
                encoding="utf-8",
                float_format="%.17g",
            )
            distribution_long = feature_sample.melt(
                id_vars=["dataset", "sample_id", "source_file", "label", "display_class"],
                value_vars=list(selected_specs.keys()),
                var_name="feature",
                value_name="value",
            )
            display_names = {
                "flow_duration": "Flow duration",
                "forward_packet_count": "Forward packet count",
                "flow_byte_rate": "Flow byte rate",
            }
            distribution_long["feature"] = distribution_long["feature"].map(display_names)
            distribution_long.to_csv(
                export_dir / "clean_feature_distribution_sample_long.csv",
                index=False,
                encoding="utf-8",
                float_format="%.17g",
            )

            # README / data dictionary.
            readme = f"""ClearFlow — Phase 2 clean-data exports
====================================

Main cleaned model dataset (local, compressed; not committed to Git):
  data/processed/cleaned_flows.csv.gz

Rows include provenance columns 'dataset' and 'source_file', feature columns,
and 'Label'. Drop 'dataset' and 'source_file' before model fitting.
Destination Port is intentionally retained. The duplicate
'Fwd Header Length.1' is excluded. Heartbleed and Infiltration are excluded
from modeling, but retained in Phase 1 raw visualizations.

CSV exports in this folder:
  preprocessing_summary_by_file.csv
    Mutually exclusive row-removal counts per input file and final row counts.
  preprocessing_summary_totals.csv
    Stage totals by source dataset and overall.
  class_counts_before_after.csv
    Raw harmonized vs cleaned class counts and percentages by source file.
  clean_class_counts_by_dataset.csv
    Cleaned class counts for charting; use this rather than counting the sample.
  feature_preprocessing_audit.csv
    Feature min/max after ambiguous-label removal and zero-variance decision.
  column_policy.csv
    Documented metadata exclusions and the Destination Port decision.
  ambiguous_feature_summary.csv / ambiguous_label_pairs.csv
    Conflicting input-vector counts and label pairs after exact raw deduplication.
  clean_feature_profile_by_dataset.csv
    Min, max, mean, and zero frequency on the full cleaned data.
  clean_visualization_sample_wide.csv
    Deterministic, class-covered sample with all retained features.
  clean_feature_distribution_sample_wide.csv / _long.csv
    Sample for comparing selected feature distributions to Phase 1.

Sample note:
  The visualization sample is capped per class per input chunk to make
  minority classes visible. It is not prevalence-representative; do not use it
  to calculate class counts. Use the clean class-count table instead.

Methodology note:
  Exact duplicates are identified from all original row fields within each
  dataset before metadata is removed. After harmonization, complete numeric
  feature vectors associated with multiple labels are excluded as ambiguous.
  No balancing, train/test splitting, or scaling is performed in Phase 2.
"""
            (export_dir / "README.txt").write_text(readme, encoding="utf-8")

            # Also save final candidate/retained feature list.
            pd.DataFrame({
                "feature": feature_columns,
                "retained_after_zero_variance_filter": [f in retained_features for f in feature_columns],
                "decision": [
                    "retained" if f in retained_features else "removed: zero variance"
                    for f in feature_columns
                ],
            }).to_csv(export_dir / "final_feature_list.csv", index=False)

            print("\nPreprocessing complete.")
            print(f"Clean dataset: {clean_path}")
            print(f"BI exports: {export_dir}")
            print(f"Raw candidate features: {len(feature_columns)}")
            print(f"Retained features: {len(retained_features)}")
            print(f"Zero-variance features removed: {len(constant_features)}")
            print(f"Final rows: {int(summary_by_file['final_rows'].sum()):,}")

            return {
                "summary_by_file": summary_by_file,
                "summary_totals": totals,
                "feature_audit": feature_audit,
                "conflict_pairs": conflict_pairs,
                "clean_class_counts": clean_counts_dataset,
                "clean_path": clean_path,
                "export_dir": export_dir,
                "feature_columns": retained_features,
            }

        finally:
            connection.close()
