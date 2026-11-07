import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.qubo_project.preprocessing import fit_normalize, main


class FitNormalizeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.input_path = self.directory / "input.csv"
        self.normalized_path = self.directory / "normalized.csv"
        self.statistics_path = self.directory / "statistics.json"

    def write_csv(self, rows: list[list[str]]) -> None:
        with self.input_path.open("w", encoding="utf-8", newline="") as csv_file:
            csv.writer(csv_file).writerows(rows)

    def run_preprocessing(self, threshold: float = 0.05) -> dict[str, object]:
        return fit_normalize(
            str(self.input_path),
            "outcome",
            str(self.normalized_path),
            str(self.statistics_path),
            threshold,
        )

    def read_output(self) -> list[dict[str, str]]:
        with self.normalized_path.open(
            "r", encoding="utf-8", newline=""
        ) as csv_file:
            return list(csv.DictReader(csv_file))

    def test_normalizes_features_and_preserves_target(self) -> None:
        self.write_csv(
            [
                ["constant", "measurement", "outcome"],
                ["5", "1", "0"],
                ["5", "2", "1"],
                ["5", "3", "0"],
            ]
        )

        self.run_preprocessing()

        rows = self.read_output()
        self.assertEqual([row["outcome"] for row in rows], ["0", "1", "0"])
        self.assertEqual([float(row["constant"]) for row in rows], [0.0, 0.0, 0.0])
        values = [float(row["measurement"]) for row in rows]
        self.assertAlmostEqual(sum(values) / len(values), 0.0)
        self.assertAlmostEqual(sum(value * value for value in values) / len(values), 1.0)

    def test_drops_sparse_feature_and_keeps_mean_imputed_missing_value(self) -> None:
        self.write_csv(
            [
                ["sparse", "measurement", "outcome"],
                ["1", "1", "0"],
                ["0", "2", "1"],
                ["", "", "0"],
                ["0", "4", "1"],
            ]
        )

        statistics = self.run_preprocessing(threshold=0.5)

        self.assertEqual(statistics["dropped_feature_names"], ["sparse"])
        rows = self.read_output()
        self.assertEqual(list(rows[0]), ["measurement", "outcome"])
        self.assertEqual(float(rows[2]["measurement"]), 0.0)

    def test_writes_required_statistics_fields(self) -> None:
        self.write_csv(
            [
                ["feature", "outcome"],
                ["2", "0"],
                ["4", "1"],
            ]
        )

        returned_statistics = self.run_preprocessing()

        self.assertTrue(self.normalized_path.is_file())
        self.assertTrue(self.statistics_path.is_file())
        with self.statistics_path.open(encoding="utf-8") as json_file:
            saved_statistics = json.load(json_file)
        expected_fields = {
            "n_input_features",
            "n_kept_features",
            "dataset_size",
            "dataset_input_time",
            "dataset_processing_time",
            "dropped_feature_names",
        }
        self.assertEqual(set(saved_statistics), expected_fields)
        self.assertEqual(saved_statistics, returned_statistics)
        self.assertEqual(saved_statistics["dataset_size"], 2)

    def test_rejects_missing_target_column(self) -> None:
        self.write_csv([["feature", "label"], ["2", "0"]])

        with self.assertRaisesRegex(ValueError, "Target column"):
            self.run_preprocessing()

    def test_rejects_non_numeric_feature(self) -> None:
        self.write_csv(
            [
                ["feature", "outcome"],
                ["not-a-number", "0"],
            ]
        )

        with self.assertRaisesRegex(ValueError, "non-numeric"):
            self.run_preprocessing()

    def test_rejects_invalid_threshold(self) -> None:
        self.write_csv([["feature", "outcome"], ["1", "0"]])

        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            self.run_preprocessing(threshold=1.1)

    def test_rejects_non_binary_target(self) -> None:
        self.write_csv([["feature", "outcome"], ["1", "2"]])

        with self.assertRaisesRegex(ValueError, "only 0 or 1"):
            self.run_preprocessing()

    def test_command_line_interface_accepts_required_arguments(self) -> None:
        self.write_csv(
            [
                ["feature", "outcome"],
                ["1", "0"],
                ["3", "1"],
            ]
        )

        result = main(
            [
                "--input",
                str(self.input_path),
                "--target",
                "outcome",
                "--out-data",
                str(self.normalized_path),
                "--out-json",
                str(self.statistics_path),
                "--min-perc-valid",
                "0.5",
            ]
        )

        self.assertEqual(result, 0)
        self.assertTrue(self.normalized_path.is_file())
        self.assertTrue(self.statistics_path.is_file())


if __name__ == "__main__":
    unittest.main()
