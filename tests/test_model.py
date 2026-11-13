import csv
import json
from pathlib import Path

import pytest

from src.qubo_project.model import main, predict, train


def write_csv(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)


def training_rows() -> list[list[object]]:
    return [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [3, 3, 1],
        [3, 4, 1],
        [4, 3, 1],
        [4, 4, 1],
    ]


def prediction_rows() -> list[list[object]]:
    return [[0, 0, 0], [1, 1, 0], [3, 3, 1], [4, 4, 1]]


def write_training_data(path: Path) -> None:
    write_csv(path, ["left", "right", "outcome"], training_rows())


def train_model(
    directory: Path, training_path: Path, classifier: str = "random_forest"
) -> tuple[Path, Path]:
    model_path = directory / f"{classifier}.joblib"
    metrics_path = directory / f"{classifier}-training.json"
    train(
        classifier,
        str(training_path),
        "outcome",
        str(model_path),
        str(metrics_path),
        seed=19,
    )
    return model_path, metrics_path


@pytest.mark.parametrize(
    "classifier", ["random_forest", "logistic_regression", "decision_tree"]
)
def test_all_supported_classifiers_train_and_predict(
    tmp_path: Path, classifier: str
) -> None:
    training_path = tmp_path / "training.csv"
    test_path = tmp_path / "test.csv"
    write_training_data(training_path)
    model_path, metrics_path = train_model(tmp_path, training_path, classifier)
    write_csv(test_path, ["left", "right", "outcome"], prediction_rows())
    predictions_path = tmp_path / f"{classifier}-predictions.csv"
    statistics_path = tmp_path / f"{classifier}-stats.json"
    training_stats = json.loads(metrics_path.read_text(encoding="utf-8"))

    stats = predict(
        str(test_path),
        "outcome",
        str(model_path),
        str(predictions_path),
        str(statistics_path),
    )

    assert training_stats["classifier"] == classifier
    assert training_stats["n_samples"] == 8
    assert training_stats["n_features"] == 2
    assert model_path.is_file()
    assert stats["classifier"] == classifier
    assert stats["n_samples"] == 4
    assert stats["target_1_count"] == 2
    assert stats["roc_auc"] is not None


def test_training_is_deterministic_and_writes_required_statistics(
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "training.csv"
    test_path = tmp_path / "test.csv"
    write_training_data(training_path)
    first_model, first_metrics = train_model(
        tmp_path, training_path, "decision_tree"
    )
    first_stats = json.loads(first_metrics.read_text(encoding="utf-8"))
    first_predictions_path = tmp_path / "first.csv"
    write_csv(test_path, ["left", "right", "outcome"], prediction_rows())
    predict(
        str(test_path),
        "outcome",
        str(first_model),
        str(first_predictions_path),
        str(tmp_path / "first-stats.json"),
    )

    second_model = tmp_path / "second.joblib"
    train(
        "decision_tree",
        str(training_path),
        "outcome",
        str(second_model),
        str(tmp_path / "second-metrics.json"),
        seed=19,
    )
    second_predictions_path = tmp_path / "second.csv"
    predict(
        str(test_path),
        "outcome",
        str(second_model),
        str(second_predictions_path),
        str(tmp_path / "second-stats.json"),
    )

    with first_predictions_path.open(encoding="utf-8") as first_file:
        first_rows = list(csv.DictReader(first_file))
    with second_predictions_path.open(encoding="utf-8") as second_file:
        second_rows = list(csv.DictReader(second_file))
    assert first_rows == second_rows
    assert first_stats["seed"] == 19
    assert first_stats["target_1_percentage"] == 50.0
    assert "training_time" in first_stats
    assert "dataset_input_time" in first_stats


def test_predictions_csv_schema_row_numbers_and_positive_scores(
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "training.csv"
    test_path = tmp_path / "test.csv"
    write_training_data(training_path)
    model_path, _ = train_model(tmp_path, training_path, "logistic_regression")
    write_csv(test_path, ["left", "right", "outcome"], prediction_rows())
    predictions_path = tmp_path / "predictions.csv"

    predict(
        str(test_path),
        "outcome",
        str(model_path),
        str(predictions_path),
        str(tmp_path / "stats.json"),
    )

    with predictions_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        assert reader.fieldnames == ["row_n", "target", "prediction", "score"]
    assert [row["row_n"] for row in rows] == ["0", "1", "2", "3"]
    assert [row["target"] for row in rows] == ["0", "0", "1", "1"]
    assert all(0 <= float(row["score"]) <= 1 for row in rows)


def test_metrics_json_contains_class_and_confusion_matrix_fields(
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "training.csv"
    test_path = tmp_path / "test.csv"
    statistics_path = tmp_path / "classification.json"
    write_training_data(training_path)
    model_path, _ = train_model(tmp_path, training_path)
    write_csv(test_path, ["left", "right", "outcome"], prediction_rows())

    result = predict(
        str(test_path),
        "outcome",
        str(model_path),
        str(tmp_path / "predictions.csv"),
        str(statistics_path),
    )

    assert json.loads(statistics_path.read_text(encoding="utf-8")) == result
    assert result["confusion_matrix"]["labels"] == [0, 1]
    assert len(result["confusion_matrix"]["matrix"]) == 2
    for label in ("class_0", "class_1"):
        assert set(result[label]) == {"precision", "recall", "f1", "support"}


def test_single_class_test_set_has_null_roc_auc(tmp_path: Path) -> None:
    training_path = tmp_path / "training.csv"
    test_path = tmp_path / "test.csv"
    write_training_data(training_path)
    model_path, _ = train_model(tmp_path, training_path)
    write_csv(test_path, ["left", "right", "outcome"], [[0, 0, 0], [1, 0, 0]])

    statistics = predict(
        str(test_path),
        "outcome",
        str(model_path),
        str(tmp_path / "predictions.csv"),
        str(tmp_path / "stats.json"),
    )

    assert statistics["roc_auc"] is None
    json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))


def test_reorders_matching_features_and_rejects_incompatible_schema(
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "training.csv"
    test_path = tmp_path / "test.csv"
    write_training_data(training_path)
    model_path, _ = train_model(tmp_path, training_path)
    write_csv(test_path, ["right", "outcome", "left"], [[0, 0, 0], [4, 1, 4]])
    result = predict(
        str(test_path),
        "outcome",
        str(model_path),
        str(tmp_path / "reordered.csv"),
        str(tmp_path / "reordered.json"),
    )
    assert result["n_samples"] == 2

    write_csv(test_path, ["left", "extra", "outcome"], [[0, 2, 0], [4, 3, 1]])
    with pytest.raises(ValueError, match="feature schema"):
        predict(
            str(test_path),
            "outcome",
            str(model_path),
            str(tmp_path / "bad.csv"),
            str(tmp_path / "bad.json"),
        )


def test_rejects_invalid_datasets_classifier_and_path_collisions(
    tmp_path: Path,
) -> None:
    training_path = tmp_path / "training.csv"
    write_csv(training_path, ["feature", "label"], [[0, 0], [1, 1]])
    with pytest.raises(ValueError, match="Target column"):
        train(
            "random_forest",
            str(training_path),
            "outcome",
            str(tmp_path / "model.joblib"),
            str(tmp_path / "metrics.json"),
        )

    write_csv(
        training_path,
        ["feature", "outcome"],
        [[0, 0], [1, 2], [2, 0], [3, 1]],
    )
    with pytest.raises(ValueError, match="only 0 or 1"):
        train(
            "random_forest",
            str(training_path),
            "outcome",
            str(tmp_path / "model.joblib"),
            str(tmp_path / "metrics.json"),
        )

    write_training_data(training_path)
    with pytest.raises(ValueError, match="Unknown classifier"):
        train(
            "unknown",
            str(training_path),
            "outcome",
            str(tmp_path / "model.joblib"),
            str(tmp_path / "metrics.json"),
        )
    with pytest.raises(ValueError, match="paths must be different"):
        train(
            "decision_tree",
            str(training_path),
            "outcome",
            str(training_path),
            str(tmp_path / "metrics.json"),
        )


def test_training_and_prediction_cli_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    training_path = tmp_path / "training.csv"
    test_path = tmp_path / "test.csv"
    model_path = tmp_path / "cli.joblib"
    write_training_data(training_path)

    result = main(
        [
            "train",
            "--classifier",
            "decision_tree",
            "--in-reduced",
            str(training_path),
            "--target",
            "outcome",
            "--out-model",
            str(model_path),
            "--out-metrics",
            str(tmp_path / "cli-training.json"),
            "--seed",
            "42",
        ]
    )
    assert result == 0
    assert capsys.readouterr().out == "Training completed successfully.\n"

    write_csv(test_path, ["left", "right", "outcome"], prediction_rows())
    result = main(
        [
            "predict",
            "--input-testset",
            str(test_path),
            "--target",
            "outcome",
            "--model",
            str(model_path),
            "--out-predictions",
            str(tmp_path / "cli-predictions.csv"),
            "--out-stats",
            str(tmp_path / "cli-stats.json"),
        ]
    )
    assert result == 0
    assert capsys.readouterr().out == (
        "Prediction and evaluation completed successfully.\n"
    )
    assert (tmp_path / "cli-predictions.csv").is_file()
