import csv
import json
import math
from collections import Counter
from pathlib import Path

import pytest

from src.qubo_project.feature_selection import select_features
from src.qubo_project.preprocessing import fit_normalize_for_split

SAMPLE_DATASET = (
    Path(__file__).resolve().parents[1] / "data" / "sample_test_dataset.csv"
)


def test_both_target_classes_meet_minimum_proportion() -> None:
    with SAMPLE_DATASET.open(encoding="utf-8", newline="") as csv_file:
        targets = [row["target"] for row in csv.DictReader(csv_file)]

    counts = Counter(targets)
    assert set(counts) == {"0", "1"}
    assert all(count / len(targets) >= 0.10 for count in counts.values())


def test_sample_dataset_runs_through_preprocessing_and_feature_selection(
    tmp_path: Path,
) -> None:
    normalized_path = tmp_path / "normalized.csv"
    statistics_path = tmp_path / "preprocessing.json"
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    optimization_path = tmp_path / "optimization.csv"
    selection_statistics_path = tmp_path / "selection.json"

    fit_normalize_for_split(
        str(SAMPLE_DATASET),
        "target",
        str(normalized_path),
        str(statistics_path),
        percTest=0.30,
    )
    with normalized_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        normalized_rows = list(reader)
        feature_columns = [
            column for column in reader.fieldnames or [] if column != "target"
        ]
    assert len(normalized_rows) == 20
    assert feature_columns
    assert all(row["target"] in {"0", "1"} for row in normalized_rows)
    for row in normalized_rows:
        assert all(math.isfinite(float(value)) for value in row.values())
    for column in feature_columns:
        values = [float(row[column]) for row in normalized_rows[:14]]
        assert sum(values) / len(values) == pytest.approx(0.0, abs=1e-12)
        assert math.sqrt(sum(value * value for value in values) / len(values)) == (
            pytest.approx(1.0)
        )
    preprocessing_statistics = json.loads(
        statistics_path.read_text(encoding="utf-8")
    )
    assert preprocessing_statistics["normalization_training_rows"] == 14
    assert preprocessing_statistics["normalization_test_rows"] == 6

    statistics = select_features(
        str(normalized_path),
        str(train_path),
        str(test_path),
        str(optimization_path),
        str(selection_statistics_path),
        "target",
        percTest=0.30,
        alpha_computations=3,
    )

    assert statistics["training_dataset_size"] == 14
    assert statistics["test_dataset_size"] == 6
    assert len(statistics["selected_vector"]) == len(feature_columns)
    assert set(statistics["selected_vector"]) <= {0, 1}
    assert (
        abs(statistics["n_selected"] - statistics["target_k"])
        <= statistics["allowance"]
    )
    for path in (
        statistics_path,
        train_path,
        test_path,
        optimization_path,
        selection_statistics_path,
    ):
        assert path.is_file()
