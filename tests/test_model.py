import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.qubo_project.model import main, predict, train


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.training_path = self.directory / "training.csv"
        self.test_path = self.directory / "test.csv"

    def write_csv(self, path: Path, headers: list[str], rows: list[list[object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)
            writer.writerows(rows)

    def training_rows(self) -> list[list[object]]:
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

    def write_training_data(self) -> None:
        self.write_csv(self.training_path, ["left", "right", "outcome"], self.training_rows())

    def train_model(self, classifier: str = "random_forest") -> tuple[Path, Path]:
        model_path = self.directory / f"{classifier}.joblib"
        metrics_path = self.directory / f"{classifier}-training.json"
        train(
            classifier,
            str(self.training_path),
            "outcome",
            str(model_path),
            str(metrics_path),
            seed=19,
        )
        return model_path, metrics_path

    def prediction_rows(self) -> list[list[object]]:
        return [[0, 0, 0], [1, 1, 0], [3, 3, 1], [4, 4, 1]]

    def test_all_supported_classifiers_train_and_predict(self) -> None:
        self.write_training_data()
        for classifier in ("random_forest", "logistic_regression", "decision_tree"):
            with self.subTest(classifier=classifier):
                model_path, metrics_path = self.train_model(classifier)
                self.write_csv(
                    self.test_path,
                    ["left", "right", "outcome"],
                    self.prediction_rows(),
                )
                predictions_path = self.directory / f"{classifier}-predictions.csv"
                statistics_path = self.directory / f"{classifier}-stats.json"
                training_stats = json.loads(metrics_path.read_text(encoding="utf-8"))
                stats = predict(
                    str(self.test_path),
                    "outcome",
                    str(model_path),
                    str(predictions_path),
                    str(statistics_path),
                )

                self.assertEqual(training_stats["classifier"], classifier)
                self.assertEqual(training_stats["n_samples"], 8)
                self.assertEqual(training_stats["n_features"], 2)
                self.assertTrue(model_path.is_file())
                self.assertEqual(stats["classifier"], classifier)
                self.assertEqual(stats["n_samples"], 4)
                self.assertEqual(stats["target_1_count"], 2)
                self.assertIsNotNone(stats["roc_auc"])

    def test_training_is_deterministic_and_writes_required_statistics(self) -> None:
        self.write_training_data()
        first_model, first_metrics = self.train_model("decision_tree")
        first_stats = json.loads(first_metrics.read_text(encoding="utf-8"))
        first_predictions_path = self.directory / "first.csv"
        self.write_csv(self.test_path, ["left", "right", "outcome"], self.prediction_rows())
        predict(
            str(self.test_path),
            "outcome",
            str(first_model),
            str(first_predictions_path),
            str(self.directory / "first-stats.json"),
        )

        second_model = self.directory / "second.joblib"
        train(
            "decision_tree",
            str(self.training_path),
            "outcome",
            str(second_model),
            str(self.directory / "second-metrics.json"),
            seed=19,
        )
        second_predictions_path = self.directory / "second.csv"
        predict(
            str(self.test_path),
            "outcome",
            str(second_model),
            str(second_predictions_path),
            str(self.directory / "second-stats.json"),
        )

        with first_predictions_path.open(encoding="utf-8") as first_file:
            first_rows = list(csv.DictReader(first_file))
        with second_predictions_path.open(encoding="utf-8") as second_file:
            second_rows = list(csv.DictReader(second_file))
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first_stats["seed"], 19)
        self.assertEqual(first_stats["target_1_percentage"], 50.0)
        self.assertIn("training_time", first_stats)
        self.assertIn("dataset_input_time", first_stats)

    def test_predictions_csv_schema_row_numbers_and_positive_scores(self) -> None:
        self.write_training_data()
        model_path, _ = self.train_model("logistic_regression")
        self.write_csv(self.test_path, ["left", "right", "outcome"], self.prediction_rows())
        predictions_path = self.directory / "predictions.csv"

        predict(
            str(self.test_path),
            "outcome",
            str(model_path),
            str(predictions_path),
            str(self.directory / "stats.json"),
        )

        with predictions_path.open(encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, ["row_n", "target", "prediction", "score"])
        self.assertEqual([row["row_n"] for row in rows], ["0", "1", "2", "3"])
        self.assertEqual([row["target"] for row in rows], ["0", "0", "1", "1"])
        self.assertTrue(all(0 <= float(row["score"]) <= 1 for row in rows))

    def test_metrics_json_contains_class_and_confusion_matrix_fields(self) -> None:
        self.write_training_data()
        model_path, _ = self.train_model()
        self.write_csv(self.test_path, ["left", "right", "outcome"], self.prediction_rows())
        statistics_path = self.directory / "classification.json"

        result = predict(
            str(self.test_path),
            "outcome",
            str(model_path),
            str(self.directory / "predictions.csv"),
            str(statistics_path),
        )

        self.assertEqual(json.loads(statistics_path.read_text(encoding="utf-8")), result)
        self.assertEqual(result["confusion_matrix"]["labels"], [0, 1])
        self.assertEqual(len(result["confusion_matrix"]["matrix"]), 2)
        for label in ("class_0", "class_1"):
            self.assertEqual(
                set(result[label]),
                {"precision", "recall", "f1", "support"},
            )

    def test_single_class_test_set_has_null_roc_auc(self) -> None:
        self.write_training_data()
        model_path, _ = self.train_model()
        self.write_csv(
            self.test_path,
            ["left", "right", "outcome"],
            [[0, 0, 0], [1, 0, 0]],
        )

        statistics = predict(
            str(self.test_path),
            "outcome",
            str(model_path),
            str(self.directory / "predictions.csv"),
            str(self.directory / "stats.json"),
        )

        self.assertIsNone(statistics["roc_auc"])
        json.loads((self.directory / "stats.json").read_text(encoding="utf-8"))

    def test_reorders_matching_features_and_rejects_incompatible_schema(self) -> None:
        self.write_training_data()
        model_path, _ = self.train_model()
        self.write_csv(
            self.test_path,
            ["right", "outcome", "left"],
            [[0, 0, 0], [4, 1, 4]],
        )
        result = predict(
            str(self.test_path),
            "outcome",
            str(model_path),
            str(self.directory / "reordered.csv"),
            str(self.directory / "reordered.json"),
        )
        self.assertEqual(result["n_samples"], 2)

        self.write_csv(
            self.test_path,
            ["left", "extra", "outcome"],
            [[0, 2, 0], [4, 3, 1]],
        )
        with self.assertRaisesRegex(ValueError, "feature schema"):
            predict(
                str(self.test_path),
                "outcome",
                str(model_path),
                str(self.directory / "bad.csv"),
                str(self.directory / "bad.json"),
            )

    def test_rejects_invalid_datasets_classifier_and_path_collisions(self) -> None:
        self.write_csv(self.training_path, ["feature", "label"], [[0, 0], [1, 1]])
        with self.assertRaisesRegex(ValueError, "Target column"):
            train(
                "random_forest",
                str(self.training_path),
                "outcome",
                str(self.directory / "model.joblib"),
                str(self.directory / "metrics.json"),
            )

        self.write_csv(
            self.training_path,
            ["feature", "outcome"],
            [[0, 0], [1, 2], [2, 0], [3, 1]],
        )
        with self.assertRaisesRegex(ValueError, "only 0 or 1"):
            train(
                "random_forest",
                str(self.training_path),
                "outcome",
                str(self.directory / "model.joblib"),
                str(self.directory / "metrics.json"),
            )

        self.write_training_data()
        with self.assertRaisesRegex(ValueError, "Unknown classifier"):
            train(
                "unknown",
                str(self.training_path),
                "outcome",
                str(self.directory / "model.joblib"),
                str(self.directory / "metrics.json"),
            )
        with self.assertRaisesRegex(ValueError, "paths must be different"):
            train(
                "decision_tree",
                str(self.training_path),
                "outcome",
                str(self.training_path),
                str(self.directory / "metrics.json"),
            )

    def test_training_and_prediction_cli_commands(self) -> None:
        self.write_training_data()
        model_path = self.directory / "cli.joblib"
        training_output = io.StringIO()
        with redirect_stdout(training_output):
            self.assertEqual(
                main(
                    [
                        "train",
                        "--classifier",
                        "decision_tree",
                        "--in-reduced",
                        str(self.training_path),
                        "--target",
                        "outcome",
                        "--out-model",
                        str(model_path),
                        "--out-metrics",
                        str(self.directory / "cli-training.json"),
                        "--seed",
                        "42",
                    ]
                ),
                0,
            )
        self.assertEqual(training_output.getvalue(), "Training completed successfully.\n")
        self.write_csv(
            self.test_path, ["left", "right", "outcome"], self.prediction_rows()
        )
        prediction_output = io.StringIO()
        with redirect_stdout(prediction_output):
            self.assertEqual(
                main(
                    [
                        "predict",
                        "--input-testset",
                        str(self.test_path),
                        "--target",
                        "outcome",
                        "--model",
                        str(model_path),
                        "--out-predictions",
                        str(self.directory / "cli-predictions.csv"),
                        "--out-stats",
                        str(self.directory / "cli-stats.json"),
                    ]
                ),
                0,
            )
        self.assertEqual(
            prediction_output.getvalue(),
            "Prediction and evaluation completed successfully.\n",
        )
        self.assertTrue((self.directory / "cli-predictions.csv").is_file())


if __name__ == "__main__":
    unittest.main()
