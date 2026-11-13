import csv
import json
from pathlib import Path

import pytest

from src.qubo_project.preprocessing import (
    fit_normalize,
    fit_normalize_for_split,
    main,
)


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)


def run_preprocessing(
    input_path: Path,
    normalized_path: Path,
    statistics_path: Path,
    threshold: float = 0.05,
) -> dict[str, int | float | list[str]]:
    return fit_normalize(
        input_csv=str(input_path),
        target_column="outcome",
        normalized_csv=str(normalized_path),
        outInitalRes_json=str(statistics_path),
        minPercValid=threshold,
    )


def read_output(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def test_normalizes_features_and_preserves_target(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    normalized_path = tmp_path / "normalized.csv"
    statistics_path = tmp_path / "statistics.json"
    write_csv(
        input_path,
        [
            ["constant", "measurement", "outcome"],
            ["5", "1", "0"],
            ["5", "2", "1"],
            ["5", "3", "0"],
        ],
    )

    run_preprocessing(input_path, normalized_path, statistics_path)

    rows = read_output(normalized_path)
    assert [row["outcome"] for row in rows] == ["0", "1", "0"]
    assert [float(row["constant"]) for row in rows] == [0.0, 0.0, 0.0]
    values = [float(row["measurement"]) for row in rows]
    assert sum(values) / len(values) == pytest.approx(0.0)
    assert sum(value * value for value in values) / len(values) == pytest.approx(1.0)


def test_split_preprocessing_fits_only_on_training_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    normalized_path = tmp_path / "normalized.csv"
    statistics_path = tmp_path / "statistics.json"
    write_csv(
        input_path,
        [
            ["measurement", "test_only", "outcome"],
            ["1", "0", "0"],
            ["2", "0", "1"],
            ["3", "0", "0"],
            ["4", "0", "1"],
            ["5", "0", "0"],
            ["6", "0", "1"],
            ["100", "10", "0"],
            ["200", "20", "1"],
        ],
    )

    statistics = fit_normalize_for_split(
        str(input_path),
        "outcome",
        str(normalized_path),
        str(statistics_path),
        percTest=0.25,
    )

    rows = read_output(normalized_path)
    training_values = [float(row["measurement"]) for row in rows[:6]]
    assert list(rows[0]) == ["measurement", "outcome"]
    assert sum(training_values) / len(training_values) == pytest.approx(0.0)
    assert sum(value * value for value in training_values) / len(training_values) == (
        pytest.approx(1.0)
    )
    assert float(rows[6]["measurement"]) > 50
    assert statistics["dropped_feature_names"] == ["test_only"]
    assert statistics["normalization_training_rows"] == 6
    assert statistics["normalization_test_rows"] == 2
    assert statistics["normalization_test_fraction"] == 0.25
    assert json.loads(statistics_path.read_text(encoding="utf-8")) == statistics


def test_drops_sparse_feature_and_keeps_mean_imputed_missing_value(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.csv"
    normalized_path = tmp_path / "normalized.csv"
    statistics_path = tmp_path / "statistics.json"
    write_csv(
        input_path,
        [
            ["sparse", "measurement", "outcome"],
            ["1", "1", "0"],
            ["0", "2", "1"],
            ["", "", "0"],
            ["0", "4", "1"],
        ],
    )

    statistics = run_preprocessing(
        input_path, normalized_path, statistics_path, threshold=0.5
    )

    assert statistics["dropped_feature_names"] == ["sparse"]
    rows = read_output(normalized_path)
    assert list(rows[0]) == ["measurement", "outcome"]
    assert float(rows[2]["measurement"]) == 0.0


def test_writes_required_statistics_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    normalized_path = tmp_path / "normalized.csv"
    statistics_path = tmp_path / "statistics.json"
    write_csv(input_path, [["feature", "outcome"], ["2", "0"], ["4", "1"]])

    returned_statistics = run_preprocessing(
        input_path, normalized_path, statistics_path
    )

    assert normalized_path.is_file()
    assert statistics_path.is_file()
    saved_statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    assert set(saved_statistics) == {
        "n_input_features",
        "n_kept_features",
        "dataset_size",
        "dataset_input_time",
        "dataset_processing_time",
        "dropped_feature_names",
    }
    assert saved_statistics == returned_statistics
    assert saved_statistics["dataset_size"] == 2


def test_rejects_missing_target_column(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_csv(input_path, [["feature", "label"], ["2", "0"]])

    with pytest.raises(ValueError, match="Target column"):
        run_preprocessing(
            input_path, tmp_path / "normalized.csv", tmp_path / "statistics.json"
        )


def test_rejects_non_numeric_feature(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_csv(input_path, [["feature", "outcome"], ["not-a-number", "0"]])

    with pytest.raises(ValueError, match="non-numeric"):
        run_preprocessing(
            input_path, tmp_path / "normalized.csv", tmp_path / "statistics.json"
        )


def test_rejects_invalid_threshold(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_csv(input_path, [["feature", "outcome"], ["1", "0"]])

    with pytest.raises(ValueError, match="between 0 and 1"):
        run_preprocessing(
            input_path,
            tmp_path / "normalized.csv",
            tmp_path / "statistics.json",
            threshold=1.1,
        )


def test_rejects_non_binary_target(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    write_csv(input_path, [["feature", "outcome"], ["1", "2"]])

    with pytest.raises(ValueError, match="only 0 or 1"):
        run_preprocessing(
            input_path, tmp_path / "normalized.csv", tmp_path / "statistics.json"
        )


def test_command_line_interface_accepts_required_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "input.csv"
    normalized_path = tmp_path / "normalized.csv"
    statistics_path = tmp_path / "statistics.json"
    write_csv(
        input_path,
        [
            ["feature", "outcome"],
            ["1", "0"],
            ["3", "1"],
            ["4", "0"],
            ["7", "1"],
        ],
    )

    result = main(
        [
            "--input",
            str(input_path),
            "--target",
            "outcome",
            "--out-data",
            str(normalized_path),
            "--out-json",
            str(statistics_path),
            "--perc-test",
            "0.25",
            "--min-perc-valid",
            "0.5",
        ]
    )

    assert result == 0
    assert normalized_path.is_file()
    assert statistics_path.is_file()
    assert capsys.readouterr().out == "Preprocessing completed successfully.\n"
