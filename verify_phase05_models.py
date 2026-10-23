from pathlib import Path
import gc
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "exports" / "phase05_model_results"
SUMMARY = RESULTS / "experiment_summary_metadata_repaired.csv"
FEATURE_SETS = ROOT / "exports" / "phase04_feature_selection" / "feature_sets.json"
SAMPLE = ROOT / "data" / "processed" / "model_training_sample.csv.gz"


def main():
    df = pd.read_csv(SUMMARY)
    feature_sets = pd.read_json(FEATURE_SETS)
    sets = feature_sets["feature_sets"].to_dict() if "feature_sets" in feature_sets else None

    # Read feature-set metadata using the proper JSON structure.
    import json
    with open(FEATURE_SETS, encoding="utf-8") as f:
        sets = json.load(f)["feature_sets"]

    sample = pd.read_csv(SAMPLE, nrows=1, compression="gzip")

    passed, failed = [], []

    for _, row in df.iterrows():
        exp_id = row["experiment_id"]
        try:
            model_path = ROOT / str(row["model_artifact"]).replace("\\", "/")
            model = joblib.load(model_path)

            features = sets[row["feature_set_id"]]["features"]
            prediction = model.predict(sample[features])

            if len(prediction) != 1:
                raise ValueError("Unexpected prediction length")

            passed.append(exp_id)
        except Exception as e:
            failed.append((exp_id, str(e)))

        finally:
            if "model" in locals():
                del model
            gc.collect()

    print("Configurations checked:", len(df))
    print("Models loaded and predicted successfully:", len(passed))
    print("Failures:", len(failed))

    for exp_id, error in failed[:20]:
        print(f"FAILED: {exp_id}: {error}")


if __name__ == "__main__":
    main()