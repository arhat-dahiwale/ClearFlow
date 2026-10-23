
from pathlib import Path, PurePosixPath
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "exports" / "phase05_model_results"
INPUT_CSV = RESULTS_DIR / "experiment_summary.csv"
OUTPUT_CSV = RESULTS_DIR / "experiment_summary_metadata_repaired.csv"
MODEL_DIR = ROOT / "results" / "models" / "phase05"


def main():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(INPUT_CSV)

    df = pd.read_csv(INPUT_CSV)

    required = {
        "experiment_id",
        "train_sample_rows",
        "model_artifact",
    }
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    if len(df) != 200 or df["experiment_id"].nunique() != 200:
        raise ValueError("Expected exactly 200 unique configurations.")

    sizes = []
    artifact_paths = []

    for _, row in df.iterrows():
        relative = row["model_artifact"]

        if pd.isna(relative) or not str(relative).strip():
            raise ValueError(
                f"Missing artifact path: {row['experiment_id']}"
            )

        relative = str(relative).replace("\\", "/")
        parts = PurePosixPath(relative).parts

        # Prevent paths from escaping the project directory.
        if ".." in parts or PurePosixPath(relative).is_absolute():
            raise ValueError(f"Unsafe artifact path: {relative}")

        model_path = ROOT.joinpath(*parts)

        if not model_path.is_file():
            raise FileNotFoundError(
                f"Model missing for {row['experiment_id']}: {model_path}"
            )

        if model_path.parent != MODEL_DIR:
            raise ValueError(f"Unexpected model directory: {model_path}")

        sizes.append(model_path.stat().st_size)
        artifact_paths.append(relative)

    # Restore the missing aliases and actual file-size metadata.
    df["training_sample_rows"] = df["train_sample_rows"]
    df["model_artifact_path"] = artifact_paths
    df["model_size_bytes"] = sizes
    df["model_size_mb"] = [size / (1024 ** 2) for size in sizes]

    if df[
        [
            "training_sample_rows",
            "model_artifact_path",
            "model_size_bytes",
            "model_size_mb",
        ]
    ].isna().any().any():
        raise ValueError("Metadata repair left missing values.")

    # Write a separate file. Never overwrite the original summary.
    df.to_csv(OUTPUT_CSV, index=False)

    print("Metadata repair complete.")
    print("Configurations:", len(df))
    print("Verified model files:", len(sizes))
    print("Total model size (MiB):", round(sum(sizes) / 1024**2, 2))
    print("Remaining missing metadata:")
    print(
        df[
            [
                "training_sample_rows",
                "model_artifact_path",
                "model_size_bytes",
                "model_size_mb",
            ]
        ].isna().sum().to_string()
    )
    print("New CSV:", OUTPUT_CSV)
    print("Original CSV unchanged.")


if __name__ == "__main__":
    main()
