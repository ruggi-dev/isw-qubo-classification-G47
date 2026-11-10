import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.qubo_project.feature_selection import (
    _build_qubo,
    _qubo_objective,
    _rank_and_standardize,
    select_features,
)


class SelectFeaturesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.input_path = self.directory / "normalized.csv"

    def write_csv(self, rows: list[list[str]]) -> None:
        with self.input_path.open("w", encoding="utf-8", newline="") as csv_file:
            csv.writer(csv_file).writerows(rows)

    def run_selection(
        self,
        *,
        input_path: Path | None = None,
        name: str = "run",
        **overrides: object,
    ) -> dict[str, int | float | bool | list[int] | list[str]]:
        source = input_path or self.input_path
        arguments: dict[str, object] = {
            "normalized_csv": str(source),
            "reducedTrain_csv": str(self.directory / f"{name}-train.csv"),
            "reducedTest_csv": str(self.directory / f"{name}-test.csv"),
            "output_ottim_csv": str(self.directory / f"{name}-optimization.csv"),
            "output_json": str(self.directory / f"{name}-statistics.json"),
            "target_column": "outcome",
            "percTest": 0.5,
            "percSelected": 0.5,
            "allowance": 0,
            "seed": 42,
            "alpha_computations": 3,
        }
        arguments.update(overrides)
        return select_features(**arguments)  # type: ignore[arg-type]

    def read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            return list(csv.DictReader(csv_file))

    def write_correlated_dataset(self, path: Path | None = None) -> None:
        rows = [
            ["first", "duplicate", "outcome"],
            ["0", "0", "0"],
            ["0", "0", "0"],
            ["1", "1", "1"],
            ["1", "1", "1"],
            ["9", "8", "0"],
            ["8", "9", "1"],
            ["7", "6", "0"],
            ["6", "7", "1"],
        ]
        destination = path or self.input_path
        with destination.open("w", encoding="utf-8", newline="") as csv_file:
            csv.writer(csv_file).writerows(rows)

    def test_builds_specification_qubo_and_objective(self) -> None:
        target = np.array([0.8, 0.4])
        correlations = np.array([[1.0, 0.5], [0.5, 1.0]])

        matrix = _build_qubo(target, correlations, alpha=0.25)
        self.assertTrue(np.allclose(matrix, [[0.2, -0.375], [-0.375, 0.1]]))
        self.assertAlmostEqual(_qubo_objective(np.array([1, 0]), matrix), -0.2)
        self.assertAlmostEqual(_qubo_objective(np.array([1, 1]), matrix), 0.45)

    def test_spearman_ranks_ties_with_average_rank(self) -> None:
        ranks = _rank_and_standardize(np.array([1.0, 1.0, 3.0]))

        self.assertTrue(
            np.allclose(ranks, [-1 / np.sqrt(6), -1 / np.sqrt(6), 2 / np.sqrt(6)])
        )

    def test_splits_in_order_and_writes_selected_features_before_target(self) -> None:
        self.write_correlated_dataset()

        stats = self.run_selection()

        train_path = self.directory / "run-train.csv"
        test_path = self.directory / "run-test.csv"
        train_rows = self.read_rows(train_path)
        test_rows = self.read_rows(test_path)
        self.assertEqual(len(train_rows), 4)
        self.assertEqual(len(test_rows), 4)
        self.assertEqual(list(train_rows[0])[-1], "outcome")
        self.assertEqual([row["outcome"] for row in train_rows], ["0", "0", "1", "1"])
        self.assertEqual([row["outcome"] for row in test_rows], ["0", "1", "0", "1"])
        self.assertEqual(stats["training_dataset_size"], 4)
        self.assertEqual(stats["test_dataset_size"], 4)
        self.assertTrue(stats["target_reached"])

    def test_handles_chunk_boundary_and_target_not_last_in_input(self) -> None:
        with self.input_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["feature_a", "outcome", "feature_b"])
            # More than one input chunk ensures the split boundary is handled across chunks.
            for row_number in range(100_001):
                target = row_number % 2
                writer.writerow([target, target, 1 - target])

        stats = self.run_selection(
            name="chunk-boundary",
            percTest=0.5,
            percSelected=0.5,
            allowance=0,
            alpha_computations=3,
        )

        train_rows = self.read_rows(self.directory / "chunk-boundary-train.csv")
        test_rows = self.read_rows(self.directory / "chunk-boundary-test.csv")
        self.assertEqual(stats["training_dataset_size"], 50_000)
        self.assertEqual(stats["test_dataset_size"], 50_001)
        self.assertEqual(len(train_rows), 50_000)
        self.assertEqual(len(test_rows), 50_001)
        self.assertEqual(list(train_rows[0])[-1], "outcome")
        self.assertEqual(test_rows[0]["outcome"], "0")

    def test_selection_does_not_depend_on_test_rows(self) -> None:
        self.write_correlated_dataset()
        # Capture the correlation inputs so this checks leakage, not just output coincidence.
        with patch(
            "src.qubo_project.feature_selection._build_qubo", wraps=_build_qubo
        ) as original_builder:
            original = self.run_selection(name="original")
        original_correlations = original_builder.call_args_list[0].args[:2]

        changed_path = self.directory / "changed-test.csv"
        self.write_correlated_dataset(changed_path)
        with changed_path.open("r", encoding="utf-8", newline="") as csv_file:
            rows = list(csv.reader(csv_file))
        # Change only rows after the training prefix.
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
            changed = self.run_selection(input_path=changed_path, name="changed-run")
        changed_correlations = changed_builder.call_args_list[0].args[:2]

        self.assertEqual(original["selected_vector"], changed["selected_vector"])
        self.assertEqual(original["alpha"], changed["alpha"])
        for original_correlation, changed_correlation in zip(
            original_correlations, changed_correlations
        ):
            self.assertTrue(np.array_equal(original_correlation, changed_correlation))

    def test_fixed_seed_is_deterministic(self) -> None:
        self.write_correlated_dataset()
        first = self.run_selection(name="first", seed=17)
        second = self.run_selection(name="second", seed=17)

        self.assertEqual(first["selected_vector"], second["selected_vector"])
        self.assertEqual(first["alpha"], second["alpha"])

    def test_selected_vector_and_json_and_optimization_csv_shapes(self) -> None:
        self.write_correlated_dataset()
        statistics = self.run_selection()

        self.assertEqual(len(statistics["selected_vector"]), 2)
        self.assertEqual(statistics["n_selected"], sum(statistics["selected_vector"]))
        with (self.directory / "run-statistics.json").open(encoding="utf-8") as json_file:
            self.assertEqual(json.load(json_file), statistics)
        with (self.directory / "run-optimization.csv").open(
            encoding="utf-8", newline=""
        ) as csv_file:
            attempts = list(csv.DictReader(csv_file))
        self.assertGreaterEqual(len(attempts), 1)
        alphas = [float(row["alpha"]) for row in attempts]
        self.assertEqual(alphas, sorted(alphas))
        self.assertEqual(len(attempts), statistics["alpha_computations"])
        self.assertTrue(
            {"optimization_time", "n_selected", "objective_value"}
            <= set(attempts[0])
        )

    def test_alpha_attempt_limit_reports_closest_selection(self) -> None:
        self.write_correlated_dataset()

        statistics = self.run_selection(
            name="limited",
            percSelected=1.0,
            allowance=0,
            alpha_computations=1,
        )

        self.assertEqual(statistics["alpha_computations"], 1)
        self.assertEqual(statistics["alpha"], 0.0)
        self.assertFalse(statistics["target_reached"])
        self.assertEqual(statistics["n_selected"], 0)

    def test_rejects_missing_target_and_non_numeric_values(self) -> None:
        self.write_csv([["feature", "label"], ["1", "0"], ["2", "1"]])
        with self.assertRaisesRegex(ValueError, "Target column"):
            self.run_selection()

        self.write_csv(
            [["feature", "outcome"], ["1", "0"], ["bad", "1"], ["2", "0"], ["3", "1"]]
        )
        with self.assertRaisesRegex(ValueError, "numeric"):
            self.run_selection()

    def test_constant_features_are_not_selected(self) -> None:
        self.write_csv(
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
            ]
        )

        with patch(
            "src.qubo_project.feature_selection._build_qubo", wraps=_build_qubo
        ) as qubo_builder:
            statistics = self.run_selection(
                percSelected=1 / 3,
                allowance=0,
                alpha_computations=2,
            )

        self.assertEqual(statistics["selected_vector"], [0, 0, 1])
        self.assertEqual(statistics["selected_feature_names"], ["signal"])
        self.assertTrue(statistics["target_reached"])
        target_correlations, feature_correlations = qubo_builder.call_args_list[0].args[
            :2
        ]
        self.assertTrue(np.array_equal(target_correlations[:2], [0.0, 0.0]))
        self.assertTrue(np.all(feature_correlations[:2, :] == 0.0))
        self.assertTrue(np.all(feature_correlations[:, :2] == 0.0))
        train_rows = self.read_rows(self.directory / "run-train.csv")
        self.assertEqual(list(train_rows[0]), ["signal", "outcome"])

    def test_rejects_constant_training_target(self) -> None:
        self.write_csv(
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
            ]
        )

        with self.assertRaisesRegex(ValueError, "Target column.*constant"):
            self.run_selection()

    def test_rejects_invalid_parameters_and_empty_split(self) -> None:
        self.write_correlated_dataset()
        for options, message in (
            ({"percTest": 0.0}, "percTest"),
            ({"percSelected": 1.1}, "percSelected"),
            ({"allowance": -1}, "allowance"),
            ({"alpha_computations": 0}, "alpha_computations"),
            ({"percTest": 0.99}, "train/test split"),
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    self.run_selection(name=f"invalid-{message}", **options)


if __name__ == "__main__":
    unittest.main()
