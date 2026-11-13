# QUBO Classification Project

A Python application for binary classification of CSV datasets. The workflow
covers preprocessing, QUBO-based feature selection, classifier training, and
prediction evaluation. Each stage can be run from the terminal, or the complete
workflow can be managed through the GUI.

## Requirements and installation

- Python 3.11 or later
- Git (optional; the project can also be downloaded as a ZIP archive)

Clone the repository using the URL shown on GitHub, or download and extract the
ZIP archive. Open a terminal in the project root.

```bash
git clone <repository-clone-url>
cd isw-qubo-classification-G47
```

Create and activate a virtual environment, then install the dependencies.

### Windows PowerShell

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents script activation, use Command Prompt and run:

```bat
.venv\Scripts\activate.bat
```

### Linux or WSL

```bash
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On some Debian/WSL installations, you may need to install the `python3-venv`
package first. Run the following commands from the project root with the
virtual environment activated.

## Running the pipeline from the terminal

### 1. Preprocessing

The preprocessing command filters features according to the minimum proportion
of non-missing, non-zero values, then z-score normalizes the retained features.
For a train/test workflow, feature filtering and normalization statistics are
fitted on the ordered training prefix only, then applied to all rows. Missing
values in retained features are imputed with the training mean. The target
column is preserved without normalization and must contain only `0` and `1`.

To see all available options:

```powershell
python src\qubo_project\preprocessing.py --help
```

For a quick, balanced pipeline test, use the included synthetic dataset
(`data/sample_test_dataset.csv`). It has 20 rows, three numeric features, and
both target classes make up 50% of the data. The larger
`data/trial_dataset_ISW.csv` dataset is also included.

```powershell
python src\qubo_project\preprocessing.py `
  --input data\sample_test_dataset.csv `
  --target target `
  --out-data outputs\normalized.csv `
  --out-json outputs\preprocessing_stats.json `
  --perc-test 0.30 `
  --min-perc-valid 0.05
```

In WSL/Linux, use forward slashes:

```bash
python src/qubo_project/preprocessing.py \
  --input data/trial_dataset_ISW.csv \
  --target target \
  --out-data outputs/normalized.csv \
  --out-json outputs/preprocessing_stats.json \
  --perc-test 0.30 \
  --min-perc-valid 0.05
```

`--min-perc-valid` accepts a number between `0` and `1` and defaults to `0.05`
(5%). The JSON output includes the number of retained features and the names of
the dropped features, training/test row counts, and fitted split fraction,
among other statistics. Use the same `--perc-test` value in preprocessing and
feature selection. On success, the terminal prints only a completion
confirmation; the results are saved to the specified files.

The specification-compatible `fit_normalize` Python function fits using every
row supplied to it. For leakage-safe train/test preprocessing in Python code,
use `fit_normalize_for_split` and pass the same `percTest` value later used by
`select_features`.

### 2. Feature selection and train/test split

Feature selection calculates correlations on the training prefix, runs
simulated annealing, and writes reduced training and test datasets,
optimization results, and statistics.

To see all available options:

```powershell
python src\qubo_project\feature_selection.py --help
```

Example using the normalized dataset produced in the previous step:

```powershell
python src\qubo_project\feature_selection.py `
  --in-normalized outputs\normalized.csv `
  --out-train outputs\reduced_train.csv `
  --out-test outputs\reduced_test.csv `
  --out-optimizations outputs\optimization.csv `
  --out-json outputs\feature_selection_stats.json `
  --target target `
  --perc-selected 0.20 `
  --allowance 1 `
  --perc-test 0.30 `
  --seed 42 `
  --alpha-computations 100
```

In WSL/Linux, use forward slashes and `\` line continuations:

```bash
python src/qubo_project/feature_selection.py \
  --in-normalized outputs/normalized.csv \
  --out-train outputs/reduced_train.csv \
  --out-test outputs/reduced_test.csv \
  --out-optimizations outputs/optimization.csv \
  --out-json outputs/feature_selection_stats.json \
  --target target \
  --perc-selected 0.20 \
  --allowance 1 \
  --perc-test 0.30 \
  --seed 42 \
  --alpha-computations 100
```

On success, the command prints only a completion confirmation. Statistics and
optimization results are saved to the specified files. The optional parameters
are:

- `--perc-test 0.30`: fraction of rows assigned to the test set. The split
  preserves row order.
- `--perc-selected 0.20`: desired fraction of features to select.
- `--allowance 1`: tolerance for the desired number of selected features.
- `--seed 42`: non-negative optimization seed.
- `--alpha-computations 100`: maximum number of alpha values to evaluate.

If the requested feature count is not reached within the tolerance, the closest
result is still selected and `target_reached` is set to `false` in the JSON
output.

The normalized CSV contains both partitions, but test rows do not contribute
to feature-validity filtering, imputation, or normalization statistics.
Feature selection then uses the same ordered split and computes correlations
from the training prefix only. Keep `--perc-test` identical in both commands
so the preprocessing fit rows and feature-selection training rows match.
Constant features remain in the normalized CSV but are treated as having zero
Spearman correlation and are not selected. Feature selection fails with an
error if the training target contains only one class.

### 3. Classifier training

The supported classifiers are `random_forest`, `logistic_regression`, and
`decision_tree`. To see the command options:

```powershell
python src\qubo_project\model.py --help
python src\qubo_project\model.py train --help
```

Example using Random Forest:

```powershell
python src\qubo_project\model.py train `
  --classifier random_forest `
  --in-reduced outputs\reduced_train.csv `
  --target target `
  --out-model outputs\model.joblib `
  --out-metrics outputs\training_metrics.json `
  --seed 42
```

The seed is optional and defaults to `42`. The command saves the trained model
and training statistics, then prints only a completion confirmation.

### 4. Prediction and evaluation

To see the prediction command options:

```powershell
python src\qubo_project\model.py predict --help
```

Example:

```powershell
python src\qubo_project\model.py predict `
  --input-testset outputs\reduced_test.csv `
  --target target `
  --model outputs\model.joblib `
  --out-predictions outputs\predictions.csv `
  --out-stats outputs\classification_stats.json
```

The predictions CSV contains the target, predicted class, and score (estimated
probability of class `1`). The JSON contains accuracy, precision, recall, F1,
the confusion matrix, and ROC-AUC. ROC-AUC is `null` if the test set contains
only one class. Metrics are saved to the JSON file, while the terminal prints
only a completion confirmation.

### Linux/WSL command paths

The Python commands above also work in WSL/Linux; use forward slashes in paths.
For example:

```bash
python src/qubo_project/model.py train \
  --classifier random_forest \
  --in-reduced outputs/reduced_train.csv \
  --target target \
  --out-model outputs/model.joblib \
  --out-metrics outputs/training_metrics.json \
  --seed 42
```

## Launching and using the GUI

With the virtual environment activated and dependencies installed, run this
command from the project root:

```powershell
python run_gui.py
```

The same command works in WSL/Linux. The **QUBO Classification Pipeline**
window guides you through the sections in the sidebar:

1. **Dashboard**: review the workflow and select **Start import**.
2. **Dataset**: select a CSV with **Browse**, load it with **Load dataset**, and
   choose the target column. The page shows a preview and warnings about
   missing values, constant columns, or non-numeric values. The target must be
   binary (`0`/`1`), and features used by the pipeline must be numeric.
3. **Preprocessing**: set the minimum valid proportion (default `0.05`) and
   select **Run preprocessing**. The summary shows the row count and the number
   of retained and dropped features.
4. **Feature Selection**: set the test percentage (default `0.30`), target
   feature percentage (default `0.20`), allowance (default `1`), and seed
   (default `42`), then select **Run feature selection**. The page shows the
   selected features and whether the target count was reached.
5. **Training**: choose a classifier and seed (default `42`), then select
   **Train classifier**.
6. **Prediction**: select **Run prediction** to classify and evaluate the test
   set.
7. **Results**: review the metrics and confusion matrix. **Export results**
   copies the available artifacts to a folder you choose.
8. **Settings**: change the output folder and review the log. **Beginner** mode
   resets the controls to their default values; **Advanced** mode lets you
   adjust the available controls without resetting them.

Navigate between sections using the sidebar or the **Next** buttons. Each
stage requires the preceding steps to be completed. The GUI displays a message
if an operation fails.

By default, the GUI saves results under `outputs/`:

- `outputs/preprocessing/`: normalized CSV and preprocessing statistics;
- `outputs/feature_selection/`: reduced train/test datasets, optimization CSV,
  and statistics;
- `outputs/model/`: model and training metrics;
- `outputs/evaluation/`: predictions and classification statistics.

## Tests

Pytest is included in `requirements.txt`. From the project root, install the
dependencies and run:

```powershell
python -m pip install -r requirements.txt
python -m pytest -v
```

In Linux/WSL, use `python3` instead:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -v
```
