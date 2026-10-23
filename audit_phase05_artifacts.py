from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "exports" / "phase05_model_results" / "experiment_summary.csv"
MODEL_DIR = ROOT / "results" / "models" / "phase05"

def main():
    if not CSV_PATH.exists():
        print(f"ERROR: Summary CSV not found: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)

    print("=== PHASE 05 ARTIFACT AUDIT ===")
    print("Summary rows:", len(df))
    print("Unique configurations:", df["experiment_id"].nunique())
    print("Duplicate configuration IDs:",
          int(df["experiment_id"].duplicated().sum()))

    print("\n=== MODEL FILE CHECK ===")

    existing = []
    missing = []
    mismatched_names = []
    actual_sizes = {}

    for _, row in df.iterrows():
        experiment_id = str(row["experiment_id"])

        # Prefer the artifact path from the completed notebook.
        value = row.get("model_artifact")
        if pd.isna(value) or not str(value).strip():
            value = row.get("model_artifact_path")

        if pd.isna(value) or not str(value).strip():
            missing.append(experiment_id)
            continue

        relative_path = str(value).replace("\\", "/")
        path = ROOT / relative_path

        if not path.is_file():
            missing.append(f"{experiment_id} -> {relative_path}")
            continue

        existing.append(experiment_id)
        actual_sizes[experiment_id] = path.stat().st_size

        if path.stem != experiment_id:
            mismatched_names.append(
                f"{experiment_id} -> {path.name}"
            )

    print("Model files found:", len(existing))
    print("Model files missing:", len(missing))
    print("Artifact filename mismatches:", len(mismatched_names))

    if missing:
        print("\nMissing artifacts (first 20):")
        for item in missing[:20]:
            print(" -", item)

    if mismatched_names:
        print("\nFilename mismatches (first 20):")
        for item in mismatched_names[:20]:
            print(" -", item)

    if actual_sizes:
        sizes = pd.Series(actual_sizes.values(), dtype="int64")
        print("\n=== ACTUAL FILE SIZES ===")
        print("Total size (MiB):", round(sizes.sum() / 1024**2, 2))
        print("Smallest file (MiB):", round(sizes.min() / 1024**2, 4))
        print("Largest file (MiB):", round(sizes.max() / 1024**2, 2))

    print("\n=== SUMMARY METADATA ===")
    for column in [
        "training_sample_rows",
        "train_sample_rows",
        "model_size_bytes",
        "model_size_mb",
        "model_artifact_path",
        "model_artifact",
    ]:
        if column in df.columns:
            print(f"{column}: missing values = {df[column].isna().sum()}")

    print("\n=== MODEL CONFIGURATION COUNTS ===")
    print(df.groupby("model_name")["experiment_id"].nunique().to_string())

    print("\nAudit complete. No files were modified.")

if __name__ == "__main__":
    main()