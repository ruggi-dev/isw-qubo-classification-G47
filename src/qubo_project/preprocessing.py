"""Preprocess numeric CSV datasets for the QUBO classification project.

The module exposes :func:`fit_normalize` for programmatic use and a command-line
interface for running the same preprocessing step from a terminal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Sequence

_MISSING_VALUES = {"", "na", "n/a", "nan", "null"}


def _parse_number(value: str, *, row_number: int, column_name: str) -> float | None:
    """Parse a finite numeric value, treating common empty tokens as missing.

    Args:
        value: Raw CSV cell text.
        row_number: One-based line number in the CSV, including the header.
        column_name: Name used to identify invalid data in error messages.

    Returns:
        The parsed number, or ``None`` for an empty or recognized missing token.

    Raises:
        ValueError: If the cell is not numeric or contains an infinite value.
    """
    if value.strip().casefold() in _MISSING_VALUES:
        return None

    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"Row {row_number}, column {column_name!r} contains a non-numeric value: {value!r}"
        ) from exc

    if not math.isfinite(number):
        raise ValueError(
            f"Row {row_number}, column {column_name!r} contains a non-finite value: {value!r}"
        )
    return number


def _temporary_path(destination: Path) -> Path:
    """Create a temporary file beside its destination for an atomic replacement."""
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


def fit_normalize(
    input_csv: str,
    target_column: str,
    normalized_csv: str,
    outputInitialRes_json: str,
    minPercValid: float = 0.05,
) -> dict[str, int | float | list[str]]:
    """Filter and z-score normalize the feature columns of a CSV dataset.

    The target column is validated as binary (0 or 1) and copied unchanged.
    A feature is retained when its fraction of non-missing, non-zero values
    across all data rows is greater than or equal to ``minPercValid``.
    Retained features are normalized using their population standard deviation.

    Missing values in retained features are imputed with the mean of the
    observed values in that column, so their normalized output is zero.
    Constant features are retained and normalized to zero. The project
    specification does not define these two edge cases; these choices ensure
    that retained output features are numeric.

    The CSV is read twice: the first pass validates the data and calculates
    feature statistics using bounded memory, and the second writes normalized
    output. This avoids loading potentially large verification datasets into
    memory.

    Args:
        input_csv: Path to the input CSV, which must include a header row.
        target_column: Name of the binary target column.
        normalized_csv: Destination path for the normalized CSV.
        outputInitialRes_json: Destination path for preprocessing statistics.
        minPercValid: Minimum fraction of non-missing, non-zero values required
            to retain a feature. Must be between 0 and 1, inclusive.

    Returns:
        A dictionary containing the preprocessing statistics written to JSON:
        input and retained feature counts, dataset size, input and processing
        times, and names of dropped features.

    Raises:
        ValueError: If the threshold or CSV data is invalid, the target column
            is missing, or output paths conflict.
        OSError: If an input or output file cannot be accessed.
    """
    try:
        threshold = float(minPercValid)
    except (TypeError, ValueError) as exc:
        raise ValueError("minPercValid must be a number between 0 and 1") from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("minPercValid must be a finite number between 0 and 1")
    if not target_column:
        raise ValueError("target_column must not be empty")

    input_path = Path(input_csv)
    normalized_path = Path(normalized_csv)
    statistics_path = Path(outputInitialRes_json)
    resolved_input = input_path.resolve()
    resolved_outputs = (normalized_path.resolve(), statistics_path.resolve())
    if resolved_input in resolved_outputs:
        raise ValueError("The input CSV and output files must have different paths")
    if resolved_outputs[0] == resolved_outputs[1]:
        raise ValueError("The normalized CSV and statistics JSON must have different paths")

    start_time = time.perf_counter()
    input_start = start_time
    row_count = 0
    headers: list[str]
    feature_headers: list[str]
    target_index: int
    means: list[float]
    m2_values: list[float]
    value_counts: list[int]
    nonzero_counts: list[int]

    try:
        with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
            reader = csv.reader(input_file)
            try:
                headers = next(reader)
            except StopIteration as exc:
                raise ValueError("The input CSV is empty or has no header row") from exc

            if not headers or any(not name for name in headers):
                raise ValueError("The input CSV must have non-empty column names")
            if len(set(headers)) != len(headers):
                raise ValueError("The input CSV contains duplicate column names")
            if target_column not in headers:
                raise ValueError(f"Target column {target_column!r} was not found in the CSV")

            target_index = headers.index(target_column)
            feature_headers = [
                name for index, name in enumerate(headers) if index != target_index
            ]
            feature_indices = [
                index for index in range(len(headers)) if index != target_index
            ]
            means = [0.0] * len(feature_headers)
            m2_values = [0.0] * len(feature_headers)
            value_counts = [0] * len(feature_headers)
            nonzero_counts = [0] * len(feature_headers)

            # Accumulate mean and variance online to keep memory usage bounded.
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(headers):
                    raise ValueError(
                        f"Row {row_number} has {len(row)} fields; expected {len(headers)}"
                    )

                target_value = _parse_number(
                    row[target_index], row_number=row_number, column_name=target_column
                )
                if target_value not in (0.0, 1.0):
                    raise ValueError(
                        f"Row {row_number}, target column {target_column!r} "
                        "must contain only 0 or 1"
                    )

                row_count += 1
                for feature_position, column_index in enumerate(feature_indices):
                    value = _parse_number(
                        row[column_index],
                        row_number=row_number,
                        column_name=feature_headers[feature_position],
                    )
                    if value is None:
                        continue

                    value_counts[feature_position] += 1
                    if value != 0:
                        nonzero_counts[feature_position] += 1
                    delta = value - means[feature_position]
                    means[feature_position] += delta / value_counts[feature_position]
                    m2_values[feature_position] += delta * (
                        value - means[feature_position]
                    )

        input_time = time.perf_counter() - input_start
        if row_count == 0:
            raise ValueError("The input CSV contains a header but no data rows")

        kept_positions = [
            position
            for position, count in enumerate(nonzero_counts)
            # The denominator is all dataset rows, including missing values.
            if count / row_count >= threshold
        ]
        kept_positions_set = set(kept_positions)
        dropped_feature_names = [
            name
            for position, name in enumerate(feature_headers)
            if position not in kept_positions_set
        ]
        if any(value_counts[position] == 0 for position in kept_positions):
            empty_columns = [
                feature_headers[position]
                for position in kept_positions
                if value_counts[position] == 0
            ]
            raise ValueError(
                "Retained feature columns contain no numeric values: "
                + ", ".join(empty_columns)
            )

        feature_position_by_index = {
            column_index: feature_position
            for feature_position, column_index in enumerate(feature_indices)
        }
        output_headers = [
            name
            for index, name in enumerate(headers)
            if index == target_index
            or feature_position_by_index[index] in kept_positions_set
        ]
        standard_deviations = {
            # A zero variance column is retained but represented by all zeros.
            position: math.sqrt(m2_values[position] / value_counts[position])
            if m2_values[position] > 0
            else 0.0
            for position in kept_positions
        }

        normalized_temp = _temporary_path(normalized_path)
        statistics_temp: Path | None = None
        try:
            with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
                reader = csv.reader(input_file)
                next(reader)
                with normalized_temp.open(
                    "w", encoding="utf-8", newline=""
                ) as output_file:
                    writer = csv.writer(output_file)
                    writer.writerow(output_headers)
                    for row_number, row in enumerate(reader, start=2):
                        if len(row) != len(headers):
                            raise ValueError(
                                f"Row {row_number} has {len(row)} fields; "
                                f"expected {len(headers)}"
                            )
                        output_row: list[str] = []
                        for column_index, column_name in enumerate(headers):
                            if column_index == target_index:
                                output_row.append(row[column_index])
                                continue

                            feature_position = feature_position_by_index[column_index]
                            if feature_position not in kept_positions_set:
                                continue
                            value = _parse_number(
                                row[column_index],
                                row_number=row_number,
                                column_name=column_name,
                            )
                            # Mean imputation maps missing cells to zero after z-scoring.
                            if value is None or standard_deviations[feature_position] == 0:
                                normalized_value = 0.0
                            else:
                                normalized_value = (
                                    value - means[feature_position]
                                ) / standard_deviations[feature_position]
                            output_row.append(str(normalized_value))
                        writer.writerow(output_row)

            statistics: dict[str, int | float | list[str]] = {
                "n_input_features": len(feature_headers),
                "n_kept_features": len(kept_positions),
                "dataset_size": row_count,
                "dataset_input_time": input_time,
                "dataset_processing_time": time.perf_counter() - start_time,
                "dropped_feature_names": dropped_feature_names,
            }
            statistics_temp = _temporary_path(statistics_path)
            with statistics_temp.open("w", encoding="utf-8", newline="") as json_file:
                json.dump(statistics, json_file, indent=2)
                json_file.write("\n")

            os.replace(normalized_temp, normalized_path)
            os.replace(statistics_temp, statistics_path)
            return statistics
        finally:
            for temporary_path in (normalized_temp, statistics_temp):
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
    except csv.Error as exc:
        raise ValueError(f"Could not parse input CSV: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Run preprocessing from the command line.

    The CLI exposes the same inputs as :func:`fit_normalize`; validation
    failures are reported through argparse with a non-zero process exit code.
    """
    parser = argparse.ArgumentParser(description="Normalize a numeric CSV dataset.")
    parser.add_argument("--input", required=True, help="Input CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--out-data", required=True, help="Normalized output CSV")
    parser.add_argument("--out-json", required=True, help="Preprocessing statistics JSON")
    parser.add_argument(
        "--min-perc-valid",
        type=float,
        default=0.05,
        help="Minimum proportion of non-zero values required to keep a feature (default: 0.05)",
    )
    args = parser.parse_args(argv)

    try:
        fit_normalize(
            args.input,
            args.target,
            args.out_data,
            args.out_json,
            args.min_perc_valid,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print("Preprocessing completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
