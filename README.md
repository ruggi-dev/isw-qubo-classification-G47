# QUBO Classification Project

## Quick start: dataset preprocessing

### 1. Get the project

Install Python 3.11 or later. Open the project repository on GitHub, select
**Code**, and choose **Download ZIP**. Extract the archive, then open a terminal
in the extracted project folder. Alternatively, use Git and the clone URL shown
on the repository's GitHub page:

```bash
git clone <repository-clone-url>
cd <extracted-project-folder>
```

Run the following commands from the project root. The example dataset is
`data/trial_dataset_ISW.csv`, and its target column is named `target`.

### WSL

```bash
python3 --version
python3 src/qubo_project/preprocessing.py \
  --input data/trial_dataset_ISW.csv \
  --target target \
  --out-data outputs/normalized.csv \
  --out-json outputs/preprocessing_stats.json \
  --min-perc-valid 0.05
```

Use Linux-style forward slashes in WSL paths.

### Windows PowerShell

```powershell
python --version
python src\qubo_project\preprocessing.py `
  --input data\trial_dataset_ISW.csv `
  --target target `
  --out-data outputs\normalized.csv `
  --out-json outputs\preprocessing_stats.json `
  --min-perc-valid 0.05
```

The preprocessing script uses Python's standard library and needs no additional
packages. It creates the `outputs` directory if necessary. On success, it writes
the normalized CSV and a JSON file containing preprocessing statistics.

`--min-perc-valid` is the minimum fraction of non-missing, non-zero values a
feature must have to be kept. For example, `0.05` means 5%; the target column
is not filtered or normalized.

To run the preprocessing and feature-selection tests from the project root,
install the project requirements as described below, then run:

```bash
python3 -m unittest discover -s tests -v
```

On Windows, use `python` instead of `python3` if that is the name of your Python
command.

## QUBO feature selection

Install the numerical and chunked-CSV dependencies before running this phase:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Debian-based WSL installations, install `python3-venv` first if the `venv`
module is unavailable.

`select_features` is available as a Python API. It computes absolute Spearman
correlations on the training prefix, searches over alpha with simulated
annealing, and writes reduced training/test CSVs plus optimization CSV/JSON
statistics:

```bash
PYTHONPATH=src python3 -c "from qubo_project.feature_selection import select_features; select_features('outputs/normalized.csv', 'outputs/reduced_train.csv', 'outputs/reduced_test.csv', 'outputs/optimization.csv', 'outputs/feature_selection_stats.json', 'target')"
```

The split is an ordered cut: the first `int(N * (1 - percTest))` rows are used
for training. Constant feature columns are retained in the normalized CSV but
treated as having zero Spearman correlation and are not selected. A constant
target in the training rows is an error because both classes are required.
Note that the current preprocessing specification normalizes the entire
dataset before this split, so this phase cannot prevent information from the
test rows influencing normalization.

## Classifier training and prediction

The model module supports `random_forest`, `logistic_regression`, and
`decision_tree`. Train using the reduced training CSV, then predict and evaluate
using the reduced test CSV:

```bash
python src/qubo_project/model.py train \
  --classifier random_forest \
  --in-reduced outputs/reduced_train.csv \
  --target target \
  --out-model outputs/model.joblib \
  --out-metrics outputs/training_metrics.json \
  --seed 42

python src/qubo_project/model.py predict \
  --input-testset outputs/reduced_test.csv \
  --target target \
  --model outputs/model.joblib \
  --out-predictions outputs/predictions.csv \
  --out-stats outputs/classification_stats.json
```

The prediction score is the estimated probability of class 1. ROC-AUC is
written as `null` when the test set contains only one target class.
