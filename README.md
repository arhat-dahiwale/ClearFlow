# ClearFlow

A collaborative data visualization and machine-learning project for **multiclass network intrusion detection** using CICIDS-2017 and the SDN TCP-SYN dataset.

ClearFlow compares machine-learning models across different attack classes, evaluates their predictive performance and computational costs, and uses SHAP to explain model predictions. Python handles data processing, model training, evaluation, and explainability. Tableau/Power BI is used to build visualizations from exported results.

**This is an offline machine-learning and data visualization project, not a live intrusion detection system.**

## Project Status

| Phase | Description | Status |
|---|---|---|
| Phase 01 | Raw Data Analysis | Complete |
| Phase 02 | Preprocessing | Complete |
| Phase 03 | Fixed Train/Test Split | Complete |
| Phase 04 | Feature Selection | Complete |
| Phase 05 | Model Training and Evaluation | Complete and verified |
| Phase 06 | SHAP Explainability | Next |
| Final | Interactive dashboard and project reporting | In progress |

Refer to [`HANDOFF.md`](HANDOFF.md) for phase-wise methodology, outputs, decisions, and handoff instructions.

## Prerequisites

- Windows
- Git for Windows with Git Bash
- Python 3.12
- VS Code with the Python and Jupyter extensions

## Setup

Clone the repository and enter its root directory.

Run:

```bash
bash setup.sh
```

Activate the virtual environment:

```bash
source .venv/Scripts/activate
```

In VS Code, select `.venv/Scripts/python.exe` as the Python interpreter and the corresponding environment as the Jupyter kernel.

Before running any notebook, verify that it is using the ClearFlow virtual environment.

## Datasets

Download or obtain the datasets separately. Raw data is excluded from Git and must be placed in the following directories.

**CICIDS-2017 CSV files:**

```text
Datasets/CICIDS2017/
```

**SDN TCP-SYN CSV files:**

```text
Datasets/SDN_TCP_SYN/
```

Keep the original CSV files unchanged.

Both datasets are combined into one unified modeling dataset. The SDN TCP-SYN dataset supplies the SYN Flood class, which remains separate from the CICIDS-2017 DDoS class.

## Project Structure

```text
ClearFlow/
├── Datasets/                         # Local raw datasets
├── data/
│   ├── interim/                      # Intermediate data artifacts
│   └── processed/                    # Cleaned data and fixed train/test splits
├── docs/                             # Project documentation
├── exports/
│   ├── phase01_raw/                  # Raw-data analysis exports
│   ├── phase02_clean/                # Preprocessing exports
│   ├── phase03_split/                # Split summaries and manifest
│   ├── phase04_feature_selection/    # Feature rankings and feature sets
│   ├── phase05_model_results/        # Model metrics and experiment manifest
│   ├── phase05_visualization/        # Aggregated model comparison exports
│   └── phase06_shap/                 # SHAP exports (to be generated)
├── notebooks/                        # Phase-wise analysis notebooks
├── results/
│   └── models/
│       └── phase05/                  # Saved Phase 05 model artifacts
├── src/                              # Reusable processing and experiment code
├── tests/                            # Automated tests
├── HANDOFF.md                        # Phase handoff and methodology details
├── README.md
├── requirements.txt
├── requirements-lock.txt
└── setup.sh
```

Some directories and files are generated as the pipeline runs. Refer to the relevant phase section in `HANDOFF.md` for the expected outputs.

## Project Workflow

The project follows this sequence:

1. **Raw Data Analysis:** Explore the original datasets, class distributions, data quality, duplicates, feature distributions, and correlations.
2. **Preprocessing:** Harmonize features and labels, remove invalid or conflicting records, and generate the unified cleaned dataset.
3. **Train/Test Split:** Create one fixed, group-aware training/test partition and verify that identical feature groups do not cross the split boundary.
4. **Feature Selection:** Rank the 70 candidate features using Random Forest importance, ANOVA F-test, and Decision Tree RFE. Create ten fixed feature sets.
5. **Model Training and Evaluation:** Compare the four model families across all ten feature sets and five random seeds.
6. **SHAP Explainability:** Explain the predictions of the saved models and export global and local feature-contribution summaries.
7. **Dashboard and Reporting:** Use the Python-generated CSV exports to build the final visualizations in Tableau/Power BI.

## Phase 05: Experiment Design and Results

The experiment design is fixed:

- **10 feature sets**
- **4 classifiers:** Decision Tree, Random Forest, XGBoost, and Logistic Regression
- **5 fixed seeds:** `425828`, `142707`, `692819`, `2184`, and `136548`
- **200 configurations** in total

All configurations reuse the same training sample and fixed test split. The primary evaluation metric is **Macro-F1**, with accuracy, weighted-F1, per-class metrics, confusion matrices, runtime, and memory measurements used for further analysis.

Phase 05 is complete. All 200 saved model artifacts were checked, and all 200 loaded and produced predictions successfully in the ClearFlow virtual environment.

The best individual configuration by Macro-F1 was:

```text
Configuration: all_features__xgboost__seed_2184
Macro-F1:      0.944746
Accuracy:      0.997268
Weighted-F1:   0.997485
```

These are the recorded results for that configuration on the fixed test split. Performance differs across classes, so consult the per-class metrics and confusion matrices rather than relying on aggregate scores alone.

## Important Handoff Instructions

**Do not retrain Phase 05.** The verified models already exist under:

```text
results/models/phase05/
```

Use the existing results and artifacts for the next phase.

The original Phase 05 experiment summary is:

```text
exports/phase05_model_results/experiment_summary.csv
```

For downstream work, use the metadata-repaired copy:

```text
exports/phase05_model_results/experiment_summary_metadata_repaired.csv
```

Aggregated comparison exports are under:

```text
exports/phase05_visualization/
```

**Important:** `src/model_training.py` does not match the notebook implementation that completed the 200 experiments. Its default seeds and configuration-ID format differ. Do not execute that module or rerun the Phase 05 training cells, as this could create duplicate or inconsistent runs. See `HANDOFF.md` for details.

## Phase 06: SHAP Explainability

The next phase is to explain all 200 saved model configurations using SHAP.

The agreed explanation sample is:

- 100 test rows per true class.
- 12 classes, giving 1,200 rows in total.
- The same 1,200 rows reused for every configuration.
- The fixed Phase 03 test split, existing Phase 04 feature sets, and saved Phase 05 models.

The sample is for explainability only. It must not replace full-test evaluation or be used to select a different winning configuration.

Phase 06 should produce global feature-importance summaries, class-specific contribution directions, selected local explanations, and CSV exports suitable for Tableau/Power BI.

Expected output directory:

```text
exports/phase06_shap/
```

Expected notebook:

```text
notebooks/06_shap_analysis.ipynb
```

Before processing all 200 configurations, verify the SHAP installation and run a small test on saved models from each model family. Preserve the existing model artifacts and experiment results.

For detailed implementation instructions and acceptance criteria, see the **Phase 06 — SHAP Explainability** section of `HANDOFF.md`.

## Class Definition

The modeling task uses the following 12 classes:

- Benign
- Bot
- DDoS
- FTP-Patator
- GoldenEye
- Hulk
- PortScan
- Slowhttptest
- Slowloris
- SSH-Patator
- SYN Flood
- Web Attack

The three CICIDS-2017 Web Attack subtypes—Brute Force, SQL Injection, and XSS—are combined into `Web Attack`. Heartbleed and Infiltration are excluded from the modeling dataset.

## Dependencies

`requirements.txt` specifies direct dependencies.

`requirements-lock.txt`, when present, records the resolved Python package versions used by the project.

Use the ClearFlow virtual environment for notebooks, training verification, and SHAP analysis. Record any new dependencies and their versions when adding functionality.