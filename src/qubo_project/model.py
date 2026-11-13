"""Train binary classifiers and generate predictions for reduced datasets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

_CLASSIFIERS = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
}


def _read_dataset(path: Path, target_column: str) -> pd.DataFrame:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as input_file:
            try:
                headers = next(csv.reader(input_file))
            except StopIteration as exc:
                raise ValueError(f"Dataset {str(path)!r} is empty or has no header") from exc
    except csv.Error as exc:
        raise ValueError(f"Could not parse dataset header: {exc}") from exc

    if not headers or any(not name for name in headers):
        raise ValueError("Dataset must have non-empty column names")
    if len(set(headers)) != len(headers):
        raise ValueError("Dataset contains duplicate column names")
    if target_column not in headers:
        raise ValueError(f"Target column {target_column!r} was not found in the dataset")

    try:
        dataset = pd.read_csv(path, dtype=np.float64, encoding="utf-8-sig")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset {str(path)!r} contains no data rows") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"Dataset must contain only numeric values: {exc}") from exc

    if dataset.empty:
        raise ValueError(f"Dataset {str(path)!r} contains no data rows")
    values = dataset.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Dataset contains missing or non-finite numeric values")

    targets = dataset[target_column].to_numpy()
    if not np.isin(targets, (0.0, 1.0)).all():
        raise ValueError(f"Target column {target_column!r} must contain only 0 or 1")
    return dataset


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        return Path(temporary_file.name)


def _validate_distinct_paths(paths: list[Path]) -> None:
    resolved_paths = [path.resolve() for path in paths]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("Input and output paths must be different")


def _create_classifier(classifier: str, seed: int):
    classifier_type = _CLASSIFIERS.get(classifier)
    if classifier_type is None:
        supported = ", ".join(_CLASSIFIERS)
        raise ValueError(
            f"Unknown classifier {classifier!r}; choose one of: {supported}"
        )
    if classifier == "random_forest":
        return classifier_type(n_estimators=100, random_state=seed, n_jobs=-1)
    if classifier == "logistic_regression":
        return classifier_type(random_state=seed, max_iter=1000)
    return classifier_type(random_state=seed)


def train(
    classifier: str,
    reducedTrain_csv: str,
    target_column: str,
    model_path: str,
    metrics_json: str,
    seed: int = 42,
) -> dict[str, object]:
    """Fit a supported classifier and save its model and training statistics."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    training_path = Path(reducedTrain_csv)
    saved_model_path = Path(model_path)
    metrics_path = Path(metrics_json)
    _validate_distinct_paths([training_path, saved_model_path, metrics_path])

    dataset_input_start = time.perf_counter()
    dataset = _read_dataset(training_path, target_column)
    dataset_input_time = time.perf_counter() - dataset_input_start
    feature_names = [name for name in dataset.columns if name != target_column]
    if not feature_names:
        raise ValueError("Training dataset must contain at least one feature")

    targets = dataset[target_column].to_numpy(dtype=np.int8)
    if np.unique(targets).size != 2:
        raise ValueError("Training target must contain both classes 0 and 1")
    features = dataset.loc[:, feature_names]
    estimator = _create_classifier(classifier, seed)

    training_start = time.perf_counter()
    estimator.fit(features, targets)
    training_time = time.perf_counter() - training_start

    artifact = {
        "format_version": 1,
        "classifier": classifier,
        "estimator": estimator,
        "feature_names": feature_names,
        "target_column": target_column,
    }
    statistics: dict[str, object] = {
        "classifier": classifier,
        "seed": seed,
        "training_dataset": str(training_path),
        "target_column": target_column,
        "model_path": str(saved_model_path),
        "n_samples": len(dataset),
        "n_features": len(feature_names),
        "target_1_percentage": float(targets.mean() * 100.0),
        "dataset_input_time": dataset_input_time,
        "training_time": training_time,
    }

    temporary_model = _temporary_path(saved_model_path)
    temporary_metrics = _temporary_path(metrics_path)
    try:
        joblib.dump(artifact, temporary_model)
        with temporary_metrics.open("w", encoding="utf-8", newline="") as metrics_file:
            json.dump(statistics, metrics_file, indent=2, allow_nan=False)
            metrics_file.write("\n")
        os.replace(temporary_model, saved_model_path)
        os.replace(temporary_metrics, metrics_path)
    finally:
        for temporary_path in (temporary_model, temporary_metrics):
            if temporary_path.exists():
                temporary_path.unlink()

    return statistics


def predict(
    reduced_Test_csv: str,
    target_column: str,
    model_path: str,
    predictions_csv: str,
    classif_stats_json: str,
) -> dict[str, object]:
    """Load a saved classifier, predict labels, and save metrics and scores."""
    test_path = Path(reduced_Test_csv)
    saved_model_path = Path(model_path)
    predictions_path = Path(predictions_csv)
    statistics_path = Path(classif_stats_json)
    _validate_distinct_paths(
        [test_path, saved_model_path, predictions_path, statistics_path]
    )

    dataset = _read_dataset(test_path, target_column)
    try:
        artifact = joblib.load(saved_model_path)
    except (OSError, EOFError, ValueError) as exc:
        raise ValueError(f"Could not load classifier from {str(saved_model_path)!r}: {exc}") from exc
    if (
        not isinstance(artifact, dict)
        or artifact.get("format_version") != 1
        or "estimator" not in artifact
        or "feature_names" not in artifact
        or "classifier" not in artifact
    ):
        raise ValueError("Saved model file does not contain a compatible classifier")
    if artifact.get("target_column") != target_column:
        raise ValueError(
            f"Model target column {artifact.get('target_column')!r} does not match "
            f"requested target column {target_column!r}"
        )

    feature_names = artifact["feature_names"]
    if not isinstance(feature_names, list) or not all(
        isinstance(name, str) for name in feature_names
    ):
        raise ValueError("Saved model contains an invalid feature schema")
    input_features = [name for name in dataset.columns if name != target_column]
    if set(input_features) != set(feature_names) or len(input_features) != len(feature_names):
        missing = sorted(set(feature_names) - set(input_features))
        unexpected = sorted(set(input_features) - set(feature_names))
        raise ValueError(
            f"Test feature schema does not match the model; missing={missing}, "
            f"unexpected={unexpected}"
        )

    estimator = artifact["estimator"]
    features = dataset.loc[:, feature_names]
    targets = dataset[target_column].to_numpy(dtype=np.int8)
    predictions = np.asarray(estimator.predict(features), dtype=np.int8)
    classes = np.asarray(estimator.classes_)
    positive_class_positions = np.flatnonzero(classes == 1)
    if positive_class_positions.size != 1:
        raise ValueError("Saved classifier does not contain positive class 1")
    if not hasattr(estimator, "predict_proba"):
        raise ValueError("Saved classifier does not provide positive-class probabilities")
    positive_scores = np.asarray(estimator.predict_proba(features))[
        :, int(positive_class_positions[0])
    ]
    if predictions.shape != targets.shape or not np.isin(predictions, (0, 1)).all():
        raise ValueError("Classifier returned invalid binary predictions")
    if not np.isfinite(positive_scores).all():
        raise ValueError("Classifier returned non-finite prediction scores")

    class_metrics = precision_recall_fscore_support(
        targets,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    roc_auc: float | None
    if np.unique(targets).size == 2:
        roc_auc = float(roc_auc_score(targets, positive_scores))
    else:
        roc_auc = None

    statistics: dict[str, object] = {
        "classifier": artifact["classifier"],
        "n_samples": len(dataset),
        "target_1_count": int(targets.sum()),
        "target_1_percentage": float(targets.mean() * 100.0),
        "accuracy": float(accuracy_score(targets, predictions)),
        "class_0": {
            "precision": float(class_metrics[0][0]),
            "recall": float(class_metrics[1][0]),
            "f1": float(class_metrics[2][0]),
            "support": int(class_metrics[3][0]),
        },
        "class_1": {
            "precision": float(class_metrics[0][1]),
            "recall": float(class_metrics[1][1]),
            "f1": float(class_metrics[2][1]),
            "support": int(class_metrics[3][1]),
        },
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": matrix.astype(int).tolist(),
        },
    }

    temporary_predictions = _temporary_path(predictions_path)
    temporary_statistics = _temporary_path(statistics_path)
    try:
        with temporary_predictions.open(
            "w", encoding="utf-8", newline=""
        ) as predictions_file:
            writer = csv.writer(predictions_file)
            writer.writerow(["row_n", "target", "prediction", "score"])
            for row_number, (target, prediction, score) in enumerate(
                zip(targets, predictions, positive_scores)
            ):
                writer.writerow(
                    [row_number, int(target), int(prediction), format(float(score), ".17g")]
                )
        with temporary_statistics.open(
            "w", encoding="utf-8", newline=""
        ) as statistics_file:
            json.dump(statistics, statistics_file, indent=2, allow_nan=False)
            statistics_file.write("\n")
        os.replace(temporary_predictions, predictions_path)
        os.replace(temporary_statistics, statistics_path)
    finally:
        for temporary_path in (temporary_predictions, temporary_statistics):
            if temporary_path.exists():
                temporary_path.unlink()

    return statistics


def main(argv: Sequence[str] | None = None) -> int:
    """Run classifier training or prediction from the command line."""
    parser = argparse.ArgumentParser(description="Train classifiers and predict binary labels.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save a classifier")
    train_parser.add_argument("--classifier", required=True, choices=sorted(_CLASSIFIERS))
    train_parser.add_argument("--in-reduced", required=True, help="Reduced training CSV")
    train_parser.add_argument("--target", required=True, help="Target column name")
    train_parser.add_argument("--out-model", required=True, help="Saved model path")
    train_parser.add_argument("--out-metrics", required=True, help="Training metrics JSON")
    train_parser.add_argument("--seed", type=int, default=42)

    predict_parser = subparsers.add_parser("predict", help="Predict and evaluate a test CSV")
    predict_parser.add_argument("--input-testset", required=True, help="Reduced test CSV")
    predict_parser.add_argument("--target", required=True, help="Target column name")
    predict_parser.add_argument("--model", required=True, help="Saved model path")
    predict_parser.add_argument("--out-predictions", required=True, help="Predictions CSV")
    predict_parser.add_argument("--out-stats", required=True, help="Classification metrics JSON")

    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            train(
                args.classifier,
                args.in_reduced,
                args.target,
                args.out_model,
                args.out_metrics,
                args.seed,
            )
            print("Training completed successfully.")
        else:
            predict(
                args.input_testset,
                args.target,
                args.model,
                args.out_predictions,
                args.out_stats,
            )
            print("Prediction and evaluation completed successfully.")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
