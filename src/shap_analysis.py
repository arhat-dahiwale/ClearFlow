"""ClearFlow Phase 06 - SHAP explainability for the 200 saved Phase 05 models.

No model is retrained, no split is regenerated and no feature selection is
rerun. Everything is read from the existing Phase 03/04/05 outputs.

Stages (all resumable; run from the project root):

    python src/shap_analysis.py validate   # check every input + all 200 artifacts
    python src/shap_analysis.py sample     # build the fixed 1,200-row sample
    python src/shap_analysis.py smoke      # one model per family, no outputs kept
    python src/shap_analysis.py align      # proves column order matches Phase 05 results
    python src/shap_analysis.py run        # explain all configurations (checkpointed)
    python src/shap_analysis.py export     # merge checkpoints into the BI exports
    python src/shap_analysis.py local      # local explanation plots + case table
    python src/shap_analysis.py check      # final acceptance checklist

`run` accepts --model-id / --feature-set / --limit / --force, so several
terminals can work on different model families at the same time.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

TARGET_COLUMN = "Label"
PROVENANCE_COLUMNS = ("dataset", "source_file")
SAMPLE_PER_CLASS = 100
SAMPLE_SEED = 42
ADDITIVITY_TOLERANCE = 1e-4
MODEL_FAMILIES = ("decision_tree", "random_forest", "xgboost", "logistic_regression")

# What each family is explained with. Scales differ, so magnitudes must not be
# compared across families (only within-class rankings and signs).
EXPLAINER_INFO: dict[str, dict[str, str]] = {
    "decision_tree": {
        "explainer_type": "shap.TreeExplainer (tree_path_dependent)",
        "output_scale": "class probability (predict_proba)",
        "baseline": "expected class probability from the tree's stored training coverage",
    },
    "random_forest": {
        "explainer_type": "shap.TreeExplainer (tree_path_dependent)",
        "output_scale": "class probability (predict_proba)",
        "baseline": "expected class probability from the trees' stored training coverage",
    },
    "xgboost": {
        "explainer_type": "xgboost native TreeSHAP (Booster.predict pred_contribs=True)",
        "output_scale": "raw margin / log-odds per class (before softmax)",
        "baseline": "booster bias term (expected margin from stored training coverage)",
    },
    "logistic_regression": {
        "explainer_type": "shap.LinearExplainer (interventional)",
        "output_scale": "linear predictor / logit per class on StandardScaler-scaled inputs",
        "baseline": "training-sample mean (the fitted scaler mean, i.e. 0 after scaling)",
    },
}


# --------------------------------------------------------------------------
# Paths and small I/O helpers
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def test(self) -> Path:
        return self.root / "data" / "processed" / "test.csv.gz"

    @property
    def split_manifest(self) -> Path:
        return self.root / "exports" / "phase03_split" / "split_manifest.json"

    @property
    def feature_sets(self) -> Path:
        return self.root / "exports" / "phase04_feature_selection" / "feature_sets.json"

    @property
    def summary(self) -> Path:
        return (self.root / "exports" / "phase05_model_results"
                / "experiment_summary_metadata_repaired.csv")

    @property
    def training_manifest(self) -> Path:
        return (self.root / "exports" / "phase05_model_results"
                / "model_training_manifest.json")

    @property
    def out(self) -> Path:
        return self.root / "exports" / "phase06_shap"

    @property
    def checkpoints(self) -> Path:
        return self.out / "_checkpoints"

    @property
    def local_dir(self) -> Path:
        return self.out / "local_explanations"

    @property
    def sample_csv(self) -> Path:
        return self.out / "shap_sample_features.csv"

    @property
    def sample_manifest(self) -> Path:
        return self.out / "shap_sample_manifest.json"


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    tmp.replace(path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, float_format="%.8g",
              compression="gzip" if path.name.endswith(".gz") else None)
    tmp.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
@dataclass
class Inputs:
    paths: Paths
    class_order: list[str]
    model_features: list[str]
    feature_sets: dict[str, list[str]]
    summary: pd.DataFrame  # includes resolved `artifact_file`
    expected_count: int


def load_inputs(root: str | Path, expected_count: int = 200) -> Inputs:
    """Validate the Phase 03/04/05 inputs and resolve all model artifacts."""
    p = Paths(Path(root).resolve())
    for required in (p.test, p.split_manifest, p.feature_sets, p.summary,
                     p.training_manifest):
        if not required.is_file():
            raise FileNotFoundError(required)

    split = _read_json(p.split_manifest)
    training = _read_json(p.training_manifest)
    class_order = list(split["class_order"])
    if class_order != list(training["class_order"]):
        raise ValueError("class_order differs between split and training manifests.")
    if len(class_order) != 12:
        raise ValueError(f"Expected 12 classes, found {len(class_order)}.")
    model_features = list(split["model_features"])
    if len(model_features) != 70:
        raise ValueError(f"Expected 70 model features, found {len(model_features)}.")

    raw_sets = _read_json(p.feature_sets)["feature_sets"]
    feature_sets: dict[str, list[str]] = {}
    for fid, spec in raw_sets.items():
        feats = list(spec["features"])
        if len(feats) != int(spec["feature_count"]) or len(set(feats)) != len(feats):
            raise ValueError(f"Feature set {fid} has a wrong size or duplicates.")
        if not set(feats) <= set(model_features):
            raise ValueError(f"Feature set {fid} contains non-model features.")
        if set(feats) & set(PROVENANCE_COLUMNS) or TARGET_COLUMN in feats:
            raise ValueError(f"Feature set {fid} contains a forbidden column.")
        feature_sets[fid] = feats

    summary = pd.read_csv(p.summary)
    need = {"experiment_id", "model_id", "feature_set_id", "seed",
            "feature_count", "macro_f1", "model_artifact_path"}
    if need - set(summary.columns):
        raise ValueError(f"Summary is missing columns: {sorted(need - set(summary.columns))}")
    if len(summary) != expected_count or summary["experiment_id"].nunique() != expected_count:
        raise ValueError(f"Expected {expected_count} unique configurations, "
                         f"found {len(summary)} rows / "
                         f"{summary['experiment_id'].nunique()} ids.")
    if not set(summary["model_id"]) <= set(MODEL_FAMILIES):
        raise ValueError(f"Unknown model ids: {set(summary['model_id']) - set(MODEL_FAMILIES)}")
    if not set(summary["feature_set_id"]) <= set(feature_sets):
        raise ValueError("Summary references an unknown feature set.")
    for _, r in summary.iterrows():
        if int(r["feature_count"]) != len(feature_sets[r["feature_set_id"]]):
            raise ValueError(f"feature_count mismatch for {r['experiment_id']}")

    files = []
    for _, r in summary.iterrows():
        rel = str(r["model_artifact_path"]).replace("\\", "/")
        parts = Path(rel).parts
        if ".." in parts or Path(rel).is_absolute():
            raise ValueError(f"Unsafe artifact path: {rel}")
        f = p.root.joinpath(*parts)
        if not f.is_file():
            raise FileNotFoundError(f"Model missing for {r['experiment_id']}: {f}")
        files.append(f)
    summary = summary.assign(artifact_file=files)

    header = pd.read_csv(p.test, nrows=0).columns
    missing = [c for c in model_features + [TARGET_COLUMN] if c not in header]
    if missing:
        raise ValueError(f"test.csv.gz is missing columns: {missing}")

    return Inputs(p, class_order, model_features, feature_sets, summary, expected_count)


# --------------------------------------------------------------------------
# Shared 1,200-row explanation sample
# --------------------------------------------------------------------------
def build_explanation_sample(inp: Inputs, per_class: int = SAMPLE_PER_CLASS,
                             seed: int = SAMPLE_SEED,
                             chunksize: int = 200_000) -> tuple[pd.DataFrame, dict]:
    """Deterministic stratified sample from the fixed Phase 03 test split."""
    test = inp.paths.test
    parts = []
    for chunk in pd.read_csv(test, usecols=[TARGET_COLUMN], chunksize=chunksize,
                             dtype={TARGET_COLUMN: "string"}):
        parts.append(chunk[TARGET_COLUMN].str.strip().to_numpy(dtype=object))
    labels = np.concatenate(parts)
    unknown = set(labels) - set(inp.class_order)
    if unknown:
        raise ValueError(f"Unexpected labels in test split: {sorted(unknown)}")

    rng = np.random.default_rng(seed)
    chosen: dict[str, np.ndarray] = {}
    available: dict[str, int] = {}
    for label in inp.class_order:  # fixed order => reproducible draws
        pos = np.flatnonzero(labels == label)
        available[label] = int(len(pos))
        if len(pos) < per_class:
            raise ValueError(f"Class {label} has only {len(pos)} test rows.")
        chosen[label] = np.sort(rng.choice(pos, size=per_class, replace=False))

    mask = np.zeros(len(labels), dtype=bool)
    mask[np.concatenate(list(chosen.values()))] = True

    header = pd.read_csv(test, nrows=0).columns
    cols = (inp.model_features + [TARGET_COLUMN]
            + [c for c in PROVENANCE_COLUMNS if c in header])
    picked, offset = [], 0
    for chunk in pd.read_csv(test, usecols=cols, chunksize=chunksize):
        m = mask[offset:offset + len(chunk)]
        if m.any():
            sub = chunk[m].copy()
            sub.insert(0, "test_row_position", np.flatnonzero(m) + offset)
            picked.append(sub)
        offset += len(chunk)
    if offset != len(labels):
        raise RuntimeError("Row count changed between passes over test.csv.gz.")

    sample = pd.concat(picked, ignore_index=True)
    sample[TARGET_COLUMN] = sample[TARGET_COLUMN].astype(str).str.strip()
    code = {c: i for i, c in enumerate(inp.class_order)}
    sample["class_index"] = sample[TARGET_COLUMN].map(code).astype(int)
    sample = sample.sort_values(["class_index", "test_row_position"]).reset_index(drop=True)
    sample.insert(0, "sample_id", np.arange(len(sample)))

    ident = "\n".join(f"{r.test_row_position}:{r.Label}" for r in sample.itertuples())
    manifest = {
        "phase": "06_shap_sample",
        "source_file": "data/processed/test.csv.gz",
        "test_rows_total": int(len(labels)),
        "sampling_seed": seed,
        "rows_per_class": per_class,
        "sample_rows": int(len(sample)),
        "method": "numpy.random.default_rng(seed).choice without replacement, "
                  "classes drawn in class_order, positions sorted ascending",
        "position_definition": "0-based row position among data rows of test.csv.gz",
        "class_order": inp.class_order,
        "test_rows_available_per_class": available,
        "sample_rows_per_class": sample[TARGET_COLUMN].value_counts().reindex(
            inp.class_order).astype(int).to_dict(),
        "sample_identity_sha256": hashlib.sha256(ident.encode()).hexdigest(),
        "test_positions_by_class": {k: v.tolist() for k, v in chosen.items()},
        "note": "Explanation sample only. Not for model selection; its class "
                "balance is artificial, so sample metrics are not full-test metrics.",
    }
    return sample, manifest


def load_or_build_sample(inp: Inputs, rebuild: bool = False) -> pd.DataFrame:
    p = inp.paths
    if p.sample_csv.is_file() and p.sample_manifest.is_file() and not rebuild:
        sample = pd.read_csv(p.sample_csv)
        man = _read_json(p.sample_manifest)
        ident = "\n".join(f"{r.test_row_position}:{r.Label}" for r in sample.itertuples())
        if hashlib.sha256(ident.encode()).hexdigest() != man["sample_identity_sha256"]:
            raise ValueError("Saved sample does not match its manifest.")
        check_sample(sample, inp)
        return sample
    sample, manifest = build_explanation_sample(inp)
    check_sample(sample, inp)
    _write_csv(sample, p.sample_csv)
    _write_json(manifest, p.sample_manifest)
    return sample


def check_sample(sample: pd.DataFrame, inp: Inputs) -> None:
    counts = sample[TARGET_COLUMN].value_counts()
    ok = (len(sample) == 12 * SAMPLE_PER_CLASS
          and set(counts.index) == set(inp.class_order)
          and (counts == SAMPLE_PER_CLASS).all()
          and sample["test_row_position"].is_unique)
    if not ok:
        raise ValueError("Sample must hold exactly 100 unique test rows per class.")


# --------------------------------------------------------------------------
# Explaining one model
# --------------------------------------------------------------------------
@dataclass
class Explanation:
    values: np.ndarray       # (n_rows, n_features, n_classes)
    base_values: np.ndarray  # (n_classes,)
    output: np.ndarray       # (n_rows, n_classes) on the explained scale
    proba: np.ndarray
    predicted: np.ndarray
    additivity_max_abs_error: float
    seconds: float
    names_validated: bool


def _as_3d(values, n: int, f: int, c: int) -> np.ndarray:
    """shap returns a list of (n, f) arrays in old versions, (n, f, c) in new."""
    arr = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    if arr.shape != (n, f, c):
        raise ValueError(f"Unexpected SHAP shape {arr.shape}, expected {(n, f, c)}")
    return arr.astype(np.float64, copy=False)


def check_model_alignment(model, features: list[str], n_classes: int) -> bool:
    """Hard-fail on feature-order or class-order mismatches."""
    names = getattr(model, "feature_names_in_", None)
    validated = names is not None
    if validated and list(names) != features:
        raise ValueError("Model feature_names_in_ differs from the Phase 04 feature list.")
    classes = np.asarray(model.classes_)
    if not np.array_equal(classes, np.arange(n_classes)):
        raise ValueError(f"Model classes_ {classes.tolist()} are not 0..{n_classes - 1}.")
    return validated


def explain_model(model, model_id: str, X: pd.DataFrame, n_classes: int) -> Explanation:
    t0 = time.perf_counter()
    n, f = X.shape
    names_ok = check_model_alignment(model, list(X.columns), n_classes)
    # Phase 05 fitted some models on plain arrays (no column names). Feed such
    # models arrays too: same numbers, and no feature-name warnings.
    Xm = X if names_ok else X.to_numpy()
    proba = np.asarray(model.predict_proba(Xm))
    predicted = np.asarray(model.predict(Xm), dtype=np.int64)

    if model_id in ("decision_tree", "random_forest"):
        import shap
        explainer = shap.TreeExplainer(model)
        values = _as_3d(explainer.shap_values(Xm, check_additivity=False), n, f, n_classes)
        base = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)
        output = proba

    elif model_id == "xgboost":
        # shap's TreeExplainer cannot parse the vector base_score that
        # XGBoost >= 3 writes for multiclass models (shap 0.46 - 0.49 raise a
        # ValueError). XGBoost's own exact TreeSHAP gives the same quantity.
        import xgboost as xgb
        booster = model.get_booster()
        if booster.feature_names is not None and list(booster.feature_names) != list(X.columns):
            raise ValueError("Booster feature names differ from the Phase 04 feature list.")
        dm = xgb.DMatrix(Xm, feature_names=booster.feature_names)
        contribs = np.asarray(booster.predict(dm, pred_contribs=True))  # (n, c, f+1)
        output = np.asarray(booster.predict(dm, output_margin=True))    # (n, c)
        if contribs.shape != (n, n_classes, f + 1):
            raise ValueError(f"Unexpected pred_contribs shape {contribs.shape}")
        values = np.transpose(contribs[:, :, :f], (0, 2, 1)).astype(np.float64)
        bias = contribs[:, :, f].astype(np.float64)
        if np.abs(bias - bias[0]).max() > 1e-6:
            raise ValueError("XGBoost bias term is not constant across rows.")
        base = bias[0]

    elif model_id == "logistic_regression":
        import shap
        scaler, clf = model.named_steps["scaler"], model.named_steps["classifier"]
        Xs = scaler.transform(Xm)  # fitted scaler only - never refit here
        # The scaler was fitted on the Phase 05 training sample, so that
        # sample's mean is exactly 0 after scaling: it is the SHAP baseline.
        explainer = shap.LinearExplainer(clf, np.zeros((1, f)))
        values = _as_3d(explainer.shap_values(Xs), n, f, n_classes)
        base = np.asarray(explainer.expected_value, dtype=np.float64).reshape(-1)
        output = clf.decision_function(Xs)

    else:
        raise ValueError(f"Unsupported model family: {model_id}")

    if base.shape != (n_classes,):
        raise ValueError(f"Unexpected base-value shape {base.shape}")
    err = float(np.abs(base[None, :] + values.sum(axis=1) - output).max())
    return Explanation(values, base, output, proba, predicted, err,
                       time.perf_counter() - t0, names_ok)


# --------------------------------------------------------------------------
# Aggregation of one configuration
# --------------------------------------------------------------------------
def summarize_class_features(values: np.ndarray, features: list[str],
                             class_order: list[str], y_codes: np.ndarray) -> pd.DataFrame:
    """One row per explained class x feature, keeping contribution direction."""
    mean_abs = np.abs(values).mean(0)
    mean_signed = values.mean(0)
    mean_pos = np.clip(values, 0, None).mean(0)
    mean_neg = np.clip(values, None, 0).mean(0)
    pct_pos = (values > 0).mean(0)
    blocks = []
    for ci, cls in enumerate(class_order):
        own = y_codes == ci
        s_own, s_other = values[own, :, ci], values[~own, :, ci]
        b = pd.DataFrame({
            "explained_class": cls,
            "class_index": ci,
            "feature": features,
            "mean_abs_shap_all_rows": mean_abs[:, ci],
            "mean_signed_shap_all_rows": mean_signed[:, ci],
            "mean_positive_shap_all_rows": mean_pos[:, ci],
            "mean_negative_shap_all_rows": mean_neg[:, ci],
            "pct_rows_positive_shap": pct_pos[:, ci],
            "mean_abs_shap_true_class_rows": np.abs(s_own).mean(0),
            "mean_signed_shap_true_class_rows": s_own.mean(0),
            "mean_signed_shap_other_class_rows": s_other.mean(0),
        })
        total = b["mean_abs_shap_all_rows"].sum()
        b["share_of_class_total_abs_shap"] = (
            b["mean_abs_shap_all_rows"] / total if total > 0 else 0.0)
        b["rank_abs_shap_all_rows"] = b["mean_abs_shap_all_rows"].rank(
            ascending=False, method="min").astype(int)
        b["rank_abs_shap_true_class_rows"] = b["mean_abs_shap_true_class_rows"].rank(
            ascending=False, method="min").astype(int)
        b["direction_true_class_rows"] = np.select(
            [b["mean_signed_shap_true_class_rows"] > 0,
             b["mean_signed_shap_true_class_rows"] < 0],
            ["toward_class", "away_from_class"], default="neutral")
        blocks.append(b)
    return pd.concat(blocks, ignore_index=True)


def _config_meta(row: pd.Series) -> dict[str, Any]:
    return {"experiment_id": row["experiment_id"], "model_id": row["model_id"],
            "feature_set_id": row["feature_set_id"], "seed": int(row["seed"])}


def _ck(inp: Inputs, exp_id: str, kind: str) -> Path:
    suffix = {"class_feature": "class_feature.csv.gz",
              "predictions": "predictions.csv.gz", "config": "config.json"}[kind]
    return inp.paths.checkpoints / f"{exp_id}__{suffix}"


def is_done(inp: Inputs, exp_id: str) -> bool:
    return all(_ck(inp, exp_id, k).is_file() for k in ("class_feature", "predictions", "config"))


def process_configuration(inp: Inputs, sample: pd.DataFrame, row: pd.Series,
                          strict: bool = True) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    meta = _config_meta(row)
    exp_id = meta["experiment_id"]
    features = inp.feature_sets[row["feature_set_id"]]
    X = sample[features].astype("float64")
    y = sample["class_index"].to_numpy()
    n_classes = len(inp.class_order)

    artifact = Path(row["artifact_file"])
    sha = _sha256(artifact)
    model = joblib.load(artifact)
    try:
        exp = explain_model(model, row["model_id"], X, n_classes)
    finally:
        del model
        gc.collect()

    if exp.additivity_max_abs_error > ADDITIVITY_TOLERANCE:
        msg = (f"{exp_id}: SHAP additivity error {exp.additivity_max_abs_error:.3g} "
               f"exceeds {ADDITIVITY_TOLERANCE}")
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)

    cf = summarize_class_features(exp.values, features, inp.class_order, y)
    for i, (k, v) in enumerate(meta.items()):
        cf.insert(i, k, v)

    classes = np.array(inp.class_order, dtype=object)
    idx = np.arange(len(sample))
    pred = pd.DataFrame({
        **{k: v for k, v in meta.items()},
        "sample_id": sample["sample_id"].to_numpy(),
        "test_row_position": sample["test_row_position"].to_numpy(),
        "true_label": classes[y],
        "predicted_label": classes[exp.predicted],
        "correct": exp.predicted == y,
        "predicted_probability": exp.proba[idx, exp.predicted],
        "true_class_probability": exp.proba[idx, y],
    })

    info = EXPLAINER_INFO[row["model_id"]]
    cfg = {
        **meta,
        "model_name": row.get("model_name", row["model_id"]),
        "feature_count": len(features),
        "explainer_type": info["explainer_type"],
        "output_scale": info["output_scale"],
        "baseline": info["baseline"],
        "n_rows_explained": len(sample),
        "n_classes": n_classes,
        "additivity_max_abs_error": exp.additivity_max_abs_error,
        "additivity_tolerance": ADDITIVITY_TOLERANCE,
        "additivity_passed": exp.additivity_max_abs_error <= ADDITIVITY_TOLERANCE,
        "predict_matches_probability_argmax": float(
            np.mean(exp.predicted == exp.proba.argmax(1))),
        "feature_names_validated": exp.names_validated,
        "explanation_sample_accuracy": float(accuracy_score(y, exp.predicted)),
        "explanation_sample_macro_f1": float(f1_score(
            y, exp.predicted, labels=np.arange(n_classes), average="macro", zero_division=0)),
        "explain_seconds": exp.seconds,
        "model_artifact_path": str(row["model_artifact_path"]).replace("\\", "/"),
        "model_artifact_sha256": sha,
    }
    _write_csv(cf, _ck(inp, exp_id, "class_feature"))
    _write_csv(pred, _ck(inp, exp_id, "predictions"))
    _write_json(cfg, _ck(inp, exp_id, "config"))  # written last = completion marker
    return cfg


def run_configurations(inp: Inputs, sample: pd.DataFrame, model_id: str | None = None,
                       feature_set_id: str | None = None, limit: int | None = None,
                       force: bool = False, strict: bool = True) -> list[str]:
    rows = inp.summary
    if model_id:
        rows = rows[rows["model_id"] == model_id]
    if feature_set_id:
        rows = rows[rows["feature_set_id"] == feature_set_id]
    if limit:
        rows = rows.head(limit)
    failures, total, t0 = [], len(rows), time.perf_counter()
    for i, (_, row) in enumerate(rows.iterrows(), 1):
        exp_id = row["experiment_id"]
        if is_done(inp, exp_id) and not force:
            print(f"[{i}/{total}] skip (done)  {exp_id}")
            continue
        try:
            cfg = process_configuration(inp, sample, row, strict=strict)
            print(f"[{i}/{total}] ok {cfg['explain_seconds']:7.1f}s  "
                  f"err={cfg['additivity_max_abs_error']:.1e}  {exp_id}")
        except Exception as exc:  # keep going; failures are reported at the end
            failures.append(exp_id)
            err = inp.paths.checkpoints / f"{exp_id}__error.txt"
            err.parent.mkdir(parents=True, exist_ok=True)
            err.write_text(repr(exc), encoding="utf-8")
            print(f"[{i}/{total}] FAILED {exp_id}: {exc!r}")
    print(f"Done in {(time.perf_counter() - t0) / 60:.1f} min; failures: {len(failures)}")
    return failures


def smoke_test(inp: Inputs, sample: pd.DataFrame) -> pd.DataFrame:
    """Explain one all_features model per family; nothing is written."""
    out = []
    for fam in MODEL_FAMILIES:
        sub = inp.summary[(inp.summary["model_id"] == fam)
                          & (inp.summary["feature_set_id"] == "all_features")]
        if sub.empty:
            sub = inp.summary[inp.summary["model_id"] == fam].tail(1)
        row = sub.iloc[0]
        features = inp.feature_sets[row["feature_set_id"]]
        model = joblib.load(row["artifact_file"])
        e = explain_model(model, fam, sample[features].astype("float64"), len(inp.class_order))
        n_cfg = int((inp.summary["model_id"] == fam).sum())
        out.append({
            "model_id": fam, "experiment_id": row["experiment_id"],
            "shap_shape": e.values.shape, "base_values_shape": e.base_values.shape,
            "additivity_max_abs_error": e.additivity_max_abs_error,
            "feature_names_validated": e.names_validated,
            "seconds_this_model": round(e.seconds, 1),
            "configs_in_family": n_cfg,
            "est_family_minutes_upper_bound": round(e.seconds * n_cfg / 60, 1),
        })
        del model
        gc.collect()
    return pd.DataFrame(out)


def alignment_check(inp: Inputs, sample: pd.DataFrame, max_mean_gap: float = 0.10) -> pd.DataFrame:
    """Prove the column order is right even for models fitted without names.

    A wrongly ordered input collapses per-class recall. We compare each model's
    recall on the 1,200-row sample with the recall Phase 05 recorded on the full
    test set. Right order => close (sampling noise only); wrong order => far off.
    """
    rec = pd.read_csv(inp.paths.root / "exports" / "phase05_model_results"
                      / "per_class_metrics.csv", usecols=["experiment_id", "label", "recall"])
    out = []
    for fam in MODEL_FAMILIES:
        for fset in ("all_features", "rf_top10"):
            sub = inp.summary[(inp.summary["model_id"] == fam)
                              & (inp.summary["feature_set_id"] == fset)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            f = inp.feature_sets[fset]
            model = joblib.load(row["artifact_file"])
            names_ok = check_model_alignment(model, f, len(inp.class_order))
            Xm = sample[f].astype("float64")
            pred = np.asarray(model.predict(Xm if names_ok else Xm.to_numpy()))
            y = sample["class_index"].to_numpy()
            mine = pd.Series({inp.class_order[c]: float(np.mean(pred[y == c] == c))
                              for c in range(len(inp.class_order))})
            ref = rec[rec["experiment_id"] == row["experiment_id"]].set_index("label")["recall"]
            gap = (mine - ref.reindex(mine.index)).abs()
            out.append({"experiment_id": row["experiment_id"],
                        "mean_recall_gap": round(float(gap.mean()), 4),
                        "max_recall_gap": round(float(gap.max()), 4),
                        "worst_class": gap.idxmax(),
                        "aligned": bool(gap.mean() <= max_mean_gap)})
            del model
            gc.collect()
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Export (merge checkpoints into the BI-ready files)
# --------------------------------------------------------------------------
def _versions() -> dict[str, str]:
    import sklearn
    out = {"python": platform.python_version(), "numpy": np.__version__,
           "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
           "joblib": joblib.__version__}
    for name in ("shap", "xgboost", "matplotlib"):
        try:
            out[name] = __import__(name).__version__
        except Exception:
            out[name] = "not installed"
    return out


def export_all(inp: Inputs) -> dict[str, Path]:
    p = inp.paths
    ids = inp.summary["experiment_id"].tolist()
    missing = [i for i in ids if not is_done(inp, i)]
    if missing:
        raise RuntimeError(f"{len(missing)} configurations are not finished, e.g. {missing[:5]}")

    cf = pd.concat((pd.read_csv(_ck(inp, i, "class_feature")) for i in ids), ignore_index=True)
    pr = pd.concat((pd.read_csv(_ck(inp, i, "predictions")) for i in ids), ignore_index=True)
    cfg = pd.DataFrame([_read_json(_ck(inp, i, "config")) for i in ids])

    top = (cf.groupby(["experiment_id", "feature"])["mean_abs_shap_all_rows"].sum()
             .reset_index().sort_values(["experiment_id", "mean_abs_shap_all_rows"],
                                        ascending=[True, False])
             .drop_duplicates("experiment_id")
             .rename(columns={"feature": "top_feature_summed_over_classes",
                              "mean_abs_shap_all_rows": "top_feature_summed_mean_abs_shap"}))
    ref = inp.summary[["experiment_id", "accuracy", "macro_f1", "weighted_f1"]].rename(columns={
        "accuracy": "full_test_accuracy", "macro_f1": "full_test_macro_f1",
        "weighted_f1": "full_test_weighted_f1"}) if "weighted_f1" in inp.summary else \
        inp.summary[["experiment_id", "macro_f1"]].rename(columns={"macro_f1": "full_test_macro_f1"})
    cfg = cfg.merge(top, on="experiment_id", how="left").merge(ref, on="experiment_id", how="left")

    if not cfg["additivity_passed"].all():
        raise RuntimeError("Some configurations failed the additivity check.")
    if cf["experiment_id"].nunique() != len(ids) or pr["experiment_id"].nunique() != len(ids):
        raise RuntimeError("Merged exports do not cover every configuration.")
    if not (pr.groupby("experiment_id")["sample_id"].nunique() == 12 * SAMPLE_PER_CLASS).all():
        raise RuntimeError("Some configuration did not use the full 1,200-row sample.")

    files = {
        "shap_configuration_summary.csv": cfg,
        "shap_class_feature_summary.csv": cf,
        "shap_sample_predictions.csv": pr,
    }
    for name, df in files.items():
        _write_csv(df, p.out / name)

    sample_manifest = _read_json(p.sample_manifest)
    manifest = {
        "phase": "06_shap_explainability",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "configurations_explained": len(ids),
        "sample_rows": 12 * SAMPLE_PER_CLASS,
        "sample_identity_sha256": sample_manifest["sample_identity_sha256"],
        "class_order": inp.class_order,
        "feature_sets_source": "exports/phase04_feature_selection/feature_sets.json",
        "versions": _versions(),
        "explainers_by_family": EXPLAINER_INFO,
        "additivity_tolerance": ADDITIVITY_TOLERANCE,
        "max_additivity_error_overall": float(cfg["additivity_max_abs_error"].max()),
        "artifact_sha256": dict(zip(cfg["experiment_id"], cfg["model_artifact_sha256"])),
        "caveats": [
            "SHAP scales differ by family (probability, XGBoost log-odds, logistic "
            "logit). Do not compare magnitudes across families; compare within-class "
            "rankings and contribution signs.",
            "The 1,200-row sample is class-balanced (100 per class). Its accuracy/F1 "
            "columns are not full-test metrics and must not be used to pick a winner.",
            "shap.TreeExplainer cannot parse multiclass XGBoost >= 3 models "
            "(base_score vector); XGBoost values come from Booster pred_contribs, "
            "which is exact TreeSHAP on the raw-margin scale.",
            "Tree explainers use tree_path_dependent coverage (no background data); "
            "the logistic baseline is the Phase 05 training-sample mean via the "
            "fitted scaler. No test data defined any baseline.",
            "SHAP shows what drives model output, not causation in network traffic.",
        ],
    }
    _write_json(manifest, p.out / "shap_manifest.json")
    (p.out / "README.txt").write_text(_readme(), encoding="utf-8")
    return {n: p.out / n for n in list(files) + ["shap_manifest.json", "README.txt"]}


def _readme() -> str:
    return """ClearFlow Phase 06 - SHAP exports
================================
Everything here was computed once in Python; Tableau/Power BI should only read it.
No models were retrained. All 200 Phase 05 artifacts were explained on the SAME
1,200 test rows (100 per class) listed in shap_sample_manifest.json.

Files
-----
shap_configuration_summary.csv  one row per configuration: explainer, SHAP output
    scale, additivity error, sample-only metrics (explanation_sample_*), full-test
    metrics from Phase 05 (full_test_*), top feature.
shap_class_feature_summary.csv  one row per configuration x explained class x feature.
    *_all_rows   : over all 1,200 rows (global view).
    *_true_class_rows : only rows whose true class is the explained class.
    mean_signed_* > 0 pushes the output TOWARD the explained class, < 0 AWAY.
    mean_positive_/mean_negative_ split the two directions so they cannot cancel.
shap_sample_predictions.csv     prediction + confidence per configuration x sample row.
shap_sample_features.csv        the 1,200 sampled rows (features + true label + ids).
shap_sample_manifest.json       seed, positions and counts of the sample.
shap_manifest.json              versions, explainers, scales, artifact hashes, caveats.
shap_local_cases.csv            top contributions for each local-explanation plot.
local_explanations/             waterfall PNGs for correct predictions and errors.

How to read it
--------------
* Scales differ by model family (probability / log-odds / logit). Compare feature
  RANKS and SIGNS across families, not raw magnitudes.
* share_of_class_total_abs_shap is a within-configuration, within-class share and
  is the safest quantity to put on one chart across families.
* explanation_sample_* metrics come from a class-balanced 1,200-row sample. Use
  full_test_* columns (Phase 05) for model comparison.
* SHAP describes model behaviour, not causes of attacks.
"""


# --------------------------------------------------------------------------
# Local explanations
# --------------------------------------------------------------------------
def choose_local_configs(inp: Inputs) -> pd.DataFrame:
    """Best full-test Macro-F1 run overall + best run of every other family."""
    s = inp.summary.sort_values("macro_f1", ascending=False)
    best = s.iloc[[0]]
    per_family = s.drop_duplicates("model_id")
    return pd.concat([best, per_family]).drop_duplicates("experiment_id").reset_index(drop=True)


def select_local_cases(pred: pd.DataFrame) -> list[dict]:
    """Deterministic representative correct cases and informative errors."""
    cases: list[dict] = []
    for cls in ("Benign", "Bot", "DDoS"):
        ok = pred[(pred["true_label"] == cls) & pred["correct"]]
        if ok.empty:
            continue
        med = ok["predicted_probability"].median()
        r = ok.assign(d=(ok["predicted_probability"] - med).abs()).sort_values(
            ["d", "sample_id"]).iloc[0]
        cases.append({"case_type": f"correct_{_slug(cls)}", "sample_id": int(r["sample_id"]),
                      "explain": [r["true_label"]]})
    fp = pred[(pred["true_label"] == "Benign") & (pred["predicted_label"] == "Bot")]
    used = set()
    if not fp.empty:
        r = fp.sort_values(["predicted_probability", "sample_id"],
                           ascending=[False, True]).iloc[0]
        used.add(int(r["sample_id"]))
        cases.append({"case_type": "error_benign_predicted_bot", "sample_id": int(r["sample_id"]),
                      "explain": ["Bot", "Benign"]})
    errs = pred[~pred["correct"] & ~pred["sample_id"].isin(used)]
    if not errs.empty:
        r = errs.sort_values(["predicted_probability", "sample_id"],
                             ascending=[False, True]).iloc[0]
        cases.append({"case_type": "error_most_confident",
                      "sample_id": int(r["sample_id"]),
                      "explain": [r["predicted_label"], r["true_label"]]})
    return cases


def _plot_waterfall(values, base, data, features, title, path: Path, max_display=12):
    import matplotlib.pyplot as plt
    import shap
    e = shap.Explanation(values=np.asarray(values, dtype=float), base_values=float(base),
                         data=np.asarray(data, dtype=float), feature_names=list(features))
    shap.plots.waterfall(e, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=9, y=1.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_local_explanations(inp: Inputs, sample: pd.DataFrame,
                             top_n_features: int = 15) -> pd.DataFrame:
    p = inp.paths
    code = {c: i for i, c in enumerate(inp.class_order)}
    records = []
    for _, row in choose_local_configs(inp).iterrows():
        exp_id, fam = row["experiment_id"], row["model_id"]
        if not _ck(inp, exp_id, "predictions").is_file():
            raise RuntimeError(f"Run the explanation stage first ({exp_id} missing).")
        pred = pd.read_csv(_ck(inp, exp_id, "predictions"))
        features = inp.feature_sets[row["feature_set_id"]]
        X = sample[features].astype("float64")
        model = joblib.load(row["artifact_file"])
        e = explain_model(model, fam, X, len(inp.class_order))
        del model
        gc.collect()
        info = EXPLAINER_INFO[fam]
        sub = p.local_dir / _slug(exp_id)
        for case in select_local_cases(pred):
            i = int(case["sample_id"])
            pr = pred.loc[pred["sample_id"] == i].iloc[0]
            for cls in case["explain"]:
                ci = code[cls]
                case_id = f"{exp_id}__{case['case_type']}__s{i}__{_slug(cls)}"
                png = sub / f"{case['case_type']}__s{i}__explain-{_slug(cls)}.png"
                title = (f"{exp_id} | true={pr['true_label']} pred={pr['predicted_label']} "
                         f"| explained class={cls}\nscale: {info['output_scale']}")
                _plot_waterfall(e.values[i, :, ci], e.base_values[ci], X.iloc[i].to_numpy(),
                                features, title, png)
                v = e.values[i, :, ci]
                for rank, j in enumerate(np.argsort(-np.abs(v))[:top_n_features], 1):
                    records.append({
                        "case_id": case_id, "case_type": case["case_type"],
                        "experiment_id": exp_id, "model_id": fam,
                        "sample_id": i, "test_row_position": int(pr["test_row_position"]),
                        "true_label": pr["true_label"], "predicted_label": pr["predicted_label"],
                        "explained_class": cls, "output_scale": info["output_scale"],
                        "base_value": float(e.base_values[ci]),
                        "model_output": float(e.output[i, ci]),
                        "feature": features[j], "feature_value": float(X.iloc[i, j]),
                        "shap_value": float(v[j]), "abs_rank": rank,
                        "plot_file": str(png.relative_to(p.root)).replace("\\", "/"),
                    })
    df = pd.DataFrame(records)
    _write_csv(df, p.out / "shap_local_cases.csv")
    return df


def acceptance_check(inp: Inputs) -> dict[str, bool]:
    """The Phase 06 acceptance checklist from HANDOFF.md, run on the real exports."""
    p = inp.paths
    cfg = pd.read_csv(p.out / "shap_configuration_summary.csv")
    pred = pd.read_csv(p.out / "shap_sample_predictions.csv")
    cf = pd.read_csv(p.out / "shap_class_feature_summary.csv")
    sample = pd.read_csv(p.sample_csv)
    sm = _read_json(p.sample_manifest)
    ids = set(inp.summary["experiment_id"])
    positions = set(sample["test_row_position"])
    same_rows = pred.groupby("experiment_id")["test_row_position"].apply(set).map(
        lambda s: s == positions).all()
    unchanged = all(
        _sha256(Path(r["artifact_file"])) == sha for r, sha in
        ((row, dict(zip(cfg["experiment_id"], cfg["model_artifact_sha256"]))[row["experiment_id"]])
         for _, row in inp.summary.iterrows()))
    return {
        f"all {inp.expected_count} configurations accounted for":
            set(cfg["experiment_id"]) == ids == set(cf["experiment_id"]) == set(pred["experiment_id"]),
        "same 1,200 test rows used for every configuration": bool(same_rows),
        "100 rows from each of the 12 classes":
            sorted(sm["sample_rows_per_class"]) == sorted(inp.class_order)
            and all(v == SAMPLE_PER_CLASS for v in sm["sample_rows_per_class"].values()),
        "SHAP additivity passed for every configuration": bool(cfg["additivity_passed"].all()),
        "every configuration x class x feature row exported":
            len(cf) == int((cfg["feature_count"] * len(inp.class_order)).sum()),
        "global importance + direction columns exported": {
            "mean_abs_shap_all_rows", "mean_signed_shap_true_class_rows",
            "mean_positive_shap_all_rows", "mean_negative_shap_all_rows"} <= set(cf.columns),
        "explainer type and output scale recorded": bool(
            cfg["explainer_type"].notna().all() and cfg["output_scale"].notna().all()),
        "local plots exist (correct predictions and errors)":
            len(list(p.local_dir.rglob("*.png"))) > 0
            and any(pd.read_csv(p.out / "shap_local_cases.csv")["case_type"].str.startswith("error")),
        "model artifacts unchanged since they were explained": bool(unchanged),
        "manifest and README written": (p.out / "shap_manifest.json").is_file()
            and (p.out / "README.txt").is_file(),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["validate", "sample", "smoke", "align", "run", "export", "local", "check"])
    ap.add_argument("--root", default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--model-id", choices=MODEL_FAMILIES)
    ap.add_argument("--feature-set")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--expected-count", type=int, default=200)
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    inp = load_inputs(a.root, a.expected_count)
    print(f"Inputs OK: {len(inp.summary)} configurations, "
          f"{len(inp.class_order)} classes, versions={_versions()}")
    if a.stage == "validate":
        return 0
    if a.stage == "check":
        results = acceptance_check(inp)
        for name, ok in results.items():
            print(f"{'PASS' if ok else 'FAIL'}  {name}")
        print("ALL CHECKS PASSED" if all(results.values()) else "SOME CHECKS FAILED")
        return 0 if all(results.values()) else 1
    sample = load_or_build_sample(inp, rebuild=(a.stage == "sample" and a.force))
    if a.stage == "sample":
        print(f"Sample ready: {len(sample)} rows -> {inp.paths.sample_csv}")
    elif a.stage == "smoke":
        print(smoke_test(inp, sample).to_string(index=False))
    elif a.stage == "align":
        res = alignment_check(inp, sample)
        print(res.to_string(index=False))
        if not res["aligned"].all():
            print("STOP: column order does not reproduce Phase 05 recall. Do not run.")
            return 1
        print("Column order reproduces Phase 05 recall: OK")
    elif a.stage == "run":
        failed = run_configurations(inp, sample, a.model_id, a.feature_set, a.limit, a.force)
        return 1 if failed else 0
    elif a.stage == "export":
        for name, path in export_all(inp).items():
            print("wrote", path)
    elif a.stage == "local":
        df = build_local_explanations(inp, sample)
        print(f"{df['case_id'].nunique()} local explanations written to {inp.paths.local_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
