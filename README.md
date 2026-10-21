# ClearFlow

A collaborative data visualization and machine learning project
using CICIDS-2017 and the SDN TCP-SYN dataset.

## Prerequisites

- Windows
- Git for Windows with Git Bash
- Python 3.12
- VS Code with the Python and Jupyter extensions

## Setup

Clone the repository and enter its root directory.

Run:

    bash setup.sh

Activate the environment:

    source .venv/Scripts/activate

Select `.venv/Scripts/python.exe` as the Python interpreter
and Jupyter kernel in VS Code.

## Datasets

Download or obtain the datasets separately. Raw data is excluded
from Git and must be placed in the following directories.

CICIDS-2017 CSV files:

    Datasets/CICIDS2017/

SDN TCP-SYN CSV files:

    Datasets/SDN_TCP_SYN/

Keep the original CSV files unchanged.

## Project structure

- `notebooks/` — exploratory analysis and visualizations
- `src/` — reusable processing and experiment code
- `tests/` — automated tests
- `Datasets/` — local datasets, excluded from Git
- `data/interim/` — intermediate data artifacts
- `data/processed/` — processed data artifacts
- `results/` — generated metrics, figures, and explanations
- `docs/` — project documentation

## Dependencies

`requirements.txt` specifies direct dependencies.
`requirements-lock.txt`, when present, records the resolved
Python package versions used by the project.