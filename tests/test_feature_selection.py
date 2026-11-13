import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.qubo_project.feature_selection import (
    _build_qubo,
    _qubo_objective,
    _rank_and_standardize,
    main,
    select_features,
)
from src.qubo_project.model import predict, train
from src.qubo_project.preprocessing import fit_normalize_for_split


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def count_data_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return sum(1 for _ in csv.reader(csv_file)) - 1


def write_correlated_dataset(path: Path) -> None:
    write_csv(
        path,
        [
            ["first", "duplicate", "outcome"],
            ["0", "0", "0"],
            ["0", "0", "0"],
            ["1", "1", "1"],
            ["1", "1", "1"],
            ["9", "8", "0"],
            ["8", "9", "1"],
            ["7", "6", "0"],
            ["6", "7", "1"],
        ],
    )


def run_selection(
    directory: Path,
    *,
    input_path: Path | None = None,
    name: str = "run",
    percTest: float = 0.5,
    percSelected: float = 0.5,
    allowance: int = 0,
    seed: int = 42,
    alpha_computations: int = 3,
) -> dict[str, int | float | bool | list[int] | list[str]]:
    source = input_path or directory / "normalized.csv"
    return select_features(
        str(source),
        str(directory / f"{name}-train.csv"),
        str(directory / f"{name}-test.csv"),
        str(directory / f"{name}-optimization.csv"),
        str(directory / f"{name}-statistics.json"),
        "outcome",
        percTest=percTest,
        percSelected=percSelected,
        allowance=allowance,
        seed=seed,
        alpha_computations=alpha_computations,
    )


def test_builds_specification_qubo_and_objective() -> None:
    target = np.array([0.8, 0.4])
    correlations = np.array([[1.0, 0.5], [0.5, 1.0]])

    matrix = _build_qubo(target, correlations, alpha=0.25)
    assert np.allclose(matrix, [[0.2, -0.375], [-0.375, 0.1]])
    assert _qubo_objective(np.array([1, 0]), matrix) == pytest.approx(-0.2)
    assert _qubo_objective(np.array([1, 1]), matrix) == pytest.approx(0.45)


def test_spearman_ranks_ties_with_average_rank() -> None:
    ranks = _rank_and_standardize(np.array([1.0, 1.0, 3.0]))

    assert np.allclose(ranks, [-1 / np.sqrt(6), -1 / np.sqrt(6), 2 / np.sqrt(6)])


def test_splits_in_order_and_writes_selected_features_before_target(
    tmp_path: Path,
) -> None:
    write_correlated_dataset(tmp_path / "normalized.csv")

    stats = run_selection(tmp_path)

    train_rows = read_rows(tmp_path / "run-train.csv")
    test_rows = read_rows(tmp_path / "run-test.csv")
    assert len(train_rows) == 4
    assert len(test_rows) == 4
    assert list(train_rows[0])[-1] == "outcome"
    assert [row["outcome"] for row in train_rows] == ["0", "0", "1", "1"]
    assert [row["outcome"] for row in test_rows] == ["0", "1", "0", "1"]
    assert stats["training_dataset_size"] == 4
    assert stats["test_dataset_size"] == 4
    assert stats["target_reached"]


def test_pipeline_handles_more_than_one_and_a_half_million_rows(
    tmp_path: Path,
) -> None:
    row_count = 1_500_001
    input_path = tmp_path / "input.csv"
    with input_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["feature_a", "outcome", "feature_b"])
        for row_number in range(row_count):
            target = row_number % 2
            writer.writerow([target, target, 1 - target])

    normalized_path = tmp_path / "normalized.csv"
    fit_normalize_for_split(
        str(input_path),
        "outcome",
        str(normalized_path),
        str(tmp_path / "preprocessing.json"),
        percTest=0.3,
    )
    reduced_train = tmp_path / "reduced-train.csv"
    reduced_test = tmp_path / "reduced-test.csv"
    selection_stats = select_features(
        str(normalized_path),
        str(reduced_train),
        str(reduced_test),
        str(tmp_path / "optimization.csv"),
        str(tmp_path / "selection.json"),
        "outcome",
        percTest=0.3,
        percSelected=0.5,
        allowance=0,
        seed=42,
        alpha_computations=3,
    )
    assert selection_stats["training_dataset_size"] == 1_050_000
    assert selection_stats["test_dataset_size"] == 450_001
    assert count_data_rows(reduced_train) == 1_050_000
    assert count_data_rows(reduced_test) == 450_001
    with reduced_test.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames[-1] == "outcome"
        assert next(reader)["outcome"] == "0"

    model_path = tmp_path / "model.joblib"
    training_stats = train(
        "decision_tree",
        str(reduced_train),
        "outcome",
        str(model_path),
        str(tmp_path / "training.json"),
        seed=42,
    )
    predictions_path = tmp_path / "predictions.csv"
    classification_stats = predict(
        str(reduced_test),
        "outcome",
        str(model_path),
        str(predictions_path),
        str(tmp_path / "classification.json"),
    )
    assert training_stats["n_samples"] == 1_050_000
    assert model_path.is_file()
    assert classification_stats["n_samples"] == 450_001
    assert count_data_rows(predictions_path) == 450_001


def test_selection_does_not_depend_on_test_rows(tmp_path: Path) -> None:
    input_path = tmp_path / "normalized.csv"
    write_correlated_dataset(input_path)
    with patch(
        "src.qubo_project.feature_selection._build_qubo", wraps=_build_qubo
    ) as original_builder:
        original = run_selection(tmp_path, name="original")
    original_correlations = original_builder.call_args_list[0].args[:2]

    changed_path = tmp_path / "changed-test.csv"
    write_correlated_dataset(changed_path)
    with changed_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.reader(csv_file))
    rows[5:] = [
        ["100", "-100", "0"],
        ["-100", "100", "1"],
        ["50", "-50", "0"],
        ["-50", "50", "1"],
    ]
    with changed_path.open("w", encoding="utf-8", newline="") as csv_file:
        csv.writer(csv_file).writerows(rows)
    with patch(
        "src.qubo_project.feature_selection._build_qubo", wraps=_build_qubo
    ) as changed_builder:
        changed = run_selection(
            tmp_path, input_path=changed_path, name="changed-run"
        )
    changed_correlations = changed_builder.call_args_list[0].args[:2]

    assert original["selected_vector"] == changed["selected_vector"]
    assert original["alpha"] == changed["alpha"]
    for original_correlation, changed_correlation in zip(
        original_correlations, changed_correlations
    ):
        assert np.array_equal(original_correlation, changed_correlation)


def test_fixed_seed_is_deterministic(tmp_path: Path) -> None:
    write_correlated_dataset(tmp_path / "normalized.csv")

    first = run_selection(tmp_path, name="first", seed=17)
    second = run_selection(tmp_path, name="second", seed=17)

    assert first["selected_vector"] == second["selected_vector"]
    assert first["alpha"] == second["alpha"]


def test_selected_vector_and_json_and_optimization_csv_shapes(
    tmp_path: Path,
) -> None:
    write_correlated_dataset(tmp_path / "normalized.csv")

    statistics = run_selection(tmp_path)

    selected_vector = statistics["selected_vector"]
    assert len(selected_vector) == statistics["n_features"]
    assert set(selected_vector) <= {0, 1}
    assert statistics["n_selected"] == sum(statistics["selected_vector"])
    assert (
        abs(statistics["n_selected"] - statistics["target_k"])
        <= statistics["allowance"]
    )
    with (tmp_path / "run-statistics.json").open(encoding="utf-8") as json_file:
        assert json.load(json_file) == statistics
    with (tmp_path / "run-optimization.csv").open(
        encoding="utf-8", newline=""
    ) as csv_file:
        attempts = list(csv.DictReader(csv_file))
    assert len(attempts) >= 1
    alphas = [float(row["alpha"]) for row in attempts]
    assert alphas == sorted(alphas)
    assert len(attempts) == statistics["alpha_computations"]
    assert {"optimization_time", "n_selected", "objective_value"} <= set(attempts[0])


def test_selects_approximately_twenty_percent_of_features(tmp_path: Path) -> None:
    headers = [f"feature_{index}" for index in range(10)] + ["outcome"]
    rows = [headers]
    for row_number in range(40):
        target = row_number % 2
        features = [
            str(1 + ((row_number * (index + 1) + index * 7) % 31) + target * 2)
            for index in range(10)
        ]
        rows.append([*features, str(target)])
    write_csv(tmp_path / "normalized.csv", rows)

    statistics = run_selection(
        tmp_path,
        percTest=0.3,
        percSelected=0.2,
        allowance=1,
        alpha_computations=100,
    )
    selected_vector = statistics["selected_vector"]

    assert len(selected_vector) == 10
    assert set(selected_vector) <= {0, 1}
    assert abs(statistics["n_selected"] - round(0.2 * 10)) <= 1


def test_alpha_attempt_limit_reports_closest_selection(tmp_path: Path) -> None:
    write_correlated_dataset(tmp_path / "normalized.csv")

    statistics = run_selection(
        tmp_path,
        name="limited",
        percSelected=1.0,
        allowance=0,
        alpha_computations=1,
    )

    assert statistics["alpha_computations"] == 1
    assert statistics["alpha"] == 0.0
    assert not statistics["target_reached"]
    assert statistics["n_selected"] == 0


def test_rejects_missing_target_and_non_numeric_values(tmp_path: Path) -> None:
    input_path = tmp_path / "normalized.csv"
    write_csv(input_path, [["feature", "label"], ["1", "0"], ["2", "1"]])
    with pytest.raises(ValueError, match="Target column"):
        run_selection(tmp_path)

    write_csv(
        input_path,
        [["feature", "outcome"], ["1", "0"], ["bad", "1"], ["2", "0"], ["3", "1"]],
    )
    with pytest.raises(ValueError, match="numeric"):
        run_selection(tmp_path)


def test_constant_features_are_not_selected(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "normalized.csv",
        [
            ["all_zero", "fixed_value", "signal", "outcome"],
            ["0", "7", "0", "0"],
            ["0", "7", "0", "0"],
            ["0", "7", "1", "1"],
            ["0", "7", "1", "1"],
            ["0", "7", "1", "0"],
            ["0", "7", "0", "1"],
            ["0", "7", "1", "0"],
            ["0", "7", "0", "1"],
        ],
    )

    with patch(
        "src.qubo_project.feature_selection._build_qubo", wraps=_build_qubo
    ) as qubo_builder:
        statistics = run_selection(
            tmp_path,
            percSelected=1 / 3,
            allowance=0,
            alpha_computations=2,
        )

    assert statistics["selected_vector"] == [0, 0, 1]
    assert statistics["selected_feature_names"] == ["signal"]
    assert statistics["target_reached"]
    target_correlations, feature_correlations = qubo_builder.call_args_list[0].args[
        :2
    ]
    assert np.array_equal(target_correlations[:2], [0.0, 0.0])
    assert np.all(feature_correlations[:2, :] == 0.0)
    assert np.all(feature_correlations[:, :2] == 0.0)
    train_rows = read_rows(tmp_path / "run-train.csv")
    assert list(train_rows[0]) == ["signal", "outcome"]


def test_rejects_constant_training_target(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "normalized.csv",
        [
            ["feature", "outcome"],
            ["0", "0"],
            ["1", "0"],
            ["2", "0"],
            ["3", "0"],
            ["4", "1"],
            ["5", "1"],
            ["6", "0"],
            ["7", "1"],
        ],
    )

    with pytest.raises(ValueError, match="Target column.*constant"):
        run_selection(tmp_path)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"percTest": 0.0}, "percTest"),
        ({"percSelected": 1.1}, "percSelected"),
        ({"allowance": -1}, "allowance"),
        ({"alpha_computations": 0}, "alpha_computations"),
        ({"percTest": 0.99}, "train/test split"),
    ],
)
def test_rejects_invalid_parameters_and_empty_split(
    tmp_path: Path, options: dict[str, object], message: str
) -> None:
    write_correlated_dataset(tmp_path / "normalized.csv")

    with pytest.raises(ValueError, match=message):
        run_selection(tmp_path, name=f"invalid-{message}", **options)


def test_command_line_interface_runs_and_prints_only_confirmation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "normalized.csv"
    write_correlated_dataset(input_path)
    train_path = tmp_path / "cli-train.csv"
    test_path = tmp_path / "cli-test.csv"
    optimization_path = tmp_path / "cli-optimization.csv"
    statistics_path = tmp_path / "cli-statistics.json"

    result = main(
        [
            "--in-normalized",
            str(input_path),
            "--out-train",
            str(train_path),
            "--out-test",
            str(test_path),
            "--out-optimizations",
            str(optimization_path),
            "--out-json",
            str(statistics_path),
            "--target",
            "outcome",
            "--perc-selected",
            "0.5",
            "--allowance",
            "0",
            "--perc-test",
            "0.5",
            "--seed",
            "42",
            "--alpha-computations",
            "3",
        ]
    )

    assert result == 0
    assert capsys.readouterr().out == "Feature selection completed successfully.\n"
    for path in (train_path, test_path, optimization_path, statistics_path):
        assert path.is_file()
    with statistics_path.open(encoding="utf-8") as json_file:
        assert json.load(json_file)["target_reached"]
