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

To run the preprocessing tests from the project root:

```bash
python3 -m unittest discover -s tests -v
```

On Windows, use `python` instead of `python3` if that is the name of your Python
command.
