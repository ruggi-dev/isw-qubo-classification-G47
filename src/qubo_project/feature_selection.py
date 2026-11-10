"""QUBO-based feature selection and ordered train/test split.

The implementation uses Spearman correlations computed on the training
prefix only. It uses chunked CSV parsing and temporary memory-mapped arrays so
large input datasets do not need to fit in RAM.
"""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

_CSV_CHUNK_SIZE = 50_000
_ANNEALING_SWEEPS = 50
_ANNEALING_RESTARTS = 2


def _temporary_path(destination: Path) -> Path:
    """Create a temporary file beside its destination for atomic replacement."""
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


def _read_headers(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as input_file:
            try:
                headers = next(csv.reader(input_file))
            except StopIteration as exc:
                raise ValueError("The normalized CSV is empty or has no header row") from exc
    except csv.Error as exc:
        raise ValueError(f"Could not parse normalized CSV header: {exc}") from exc

    if not headers or any(not name for name in headers):
        raise ValueError("The normalized CSV must have non-empty column names")
    if len(set(headers)) != len(headers):
        raise ValueError("The normalized CSV contains duplicate column names")
    return headers


def _rank_and_standardize(values: np.ndarray) -> np.ndarray:
    """Return centered, unit-length average ranks for one numeric column."""
    if values.size < 2:
        raise ValueError("At least two training rows are required for correlations")

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    # Spearman assigns tied values their average (midpoint) rank.
    starts = np.concatenate(
        (np.array([0], dtype=np.intp), np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1)
    )
    counts = np.diff(np.append(starts, values.size))
    sorted_ranks = np.repeat(starts + (counts + 1) / 2.0, counts)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = sorted_ranks

    ranks -= ranks.mean()
    norm = float(np.linalg.norm(ranks))
    if norm == 0:
        return np.zeros_like(ranks)
    ranks /= norm
    return ranks


def _correlation_matrices(
    raw_data: np.memmap,
    training_rows: int,
    feature_count: int,
    target_index: int,
    ranked_features_path: Path,
    target_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate absolute Spearman feature-target and feature-feature correlations."""
    # Rank only the training prefix; test rows must not affect feature selection.
    target_values = np.asarray(raw_data[:training_rows, target_index])
    if np.all(target_values == target_values[0]):
        raise ValueError(
            f"Target column {target_name!r} is constant in the training rows; "
            "feature selection requires both target classes"
        )
    target_ranks = _rank_and_standardize(target_values)
    ranked_features = np.memmap(
        ranked_features_path,
        mode="w+",
        dtype=np.float64,
        shape=(training_rows, feature_count),
    )

    feature_indices = [index for index in range(raw_data.shape[1]) if index != target_index]
    for feature_position, column_index in enumerate(feature_indices):
        feature_ranks = _rank_and_standardize(
            np.asarray(raw_data[:training_rows, column_index])
        )
        ranked_features[:, feature_position] = feature_ranks
        del feature_ranks

    ranked_features.flush()
    target_correlations = np.asarray(ranked_features.T @ target_ranks)
    feature_correlations = np.asarray(ranked_features.T @ ranked_features)
    del ranked_features

    return (
        np.clip(np.abs(target_correlations), 0.0, 1.0),
        np.clip(np.abs(feature_correlations), 0.0, 1.0),
    )


def _build_qubo(
    target_correlations: np.ndarray,
    feature_correlations: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Build the symmetric Q matrix from the specification's objective."""
    # Target influence goes on the diagonal; pairwise redundancy is off-diagonal.
    matrix = -(1.0 - alpha) * feature_correlations
    np.fill_diagonal(matrix, alpha * target_correlations)
    return matrix


def _qubo_objective(state: np.ndarray, matrix: np.ndarray) -> float:
    """Evaluate the minimization objective ``-x.T @ Q @ x``."""
    return float(-(state @ matrix @ state))


def _simulated_annealing(
    matrix: np.ndarray,
    seed: int,
    selectable_features: np.ndarray | None = None,
) -> np.ndarray:
    """Approximately maximize x.T @ Q @ x using seeded simulated annealing.

    A small dense QUBO and a bounded CPU-only run make simulated annealing a
    practical open-source algorithm choice without requiring a specialized
    solver or network access.
    """
    size = matrix.shape[0]
    if size == 0:
        return np.zeros(0, dtype=np.int8)

    if selectable_features is not None:
        selectable_indices = np.flatnonzero(selectable_features)
        if selectable_indices.size != size:
            reduced_state = np.zeros(size, dtype=np.int8)
            if selectable_indices.size:
                reduced_matrix = matrix[np.ix_(selectable_indices, selectable_indices)]
                reduced_state[selectable_indices] = _simulated_annealing(
                    reduced_matrix, seed
                )
            return reduced_state

    rng = np.random.default_rng(seed)
    if np.all(matrix == 0):
        return np.zeros(size, dtype=np.int8)
    if np.all(np.diag(matrix) >= 0) and np.all(matrix - np.diag(np.diag(matrix)) <= 0):
        if np.all(matrix - np.diag(np.diag(matrix)) == 0):
            return (np.diag(matrix) > 0).astype(np.int8)

    best_state = np.zeros(size, dtype=np.int8)
    best_energy = float(best_state @ matrix @ best_state)
    scale = max(float(np.max(np.sum(np.abs(matrix), axis=1))), 1e-12)
    temperatures = np.geomspace(scale, scale * 1e-3, _ANNEALING_SWEEPS)

    for restart in range(_ANNEALING_RESTARTS):
        state = (
            np.zeros(size, dtype=np.int8)
            if restart == 0
            else rng.integers(0, 2, size=size, dtype=np.int8)
        )
        field = matrix @ state
        energy = float(state @ field)
        if energy > best_energy:
            best_state = state.copy()
            best_energy = energy

        for temperature in temperatures:
            for index in rng.permutation(size):
                direction = 1 - 2 * int(state[index])
                # Compute the energy change for one bit flip without rebuilding x.T @ Q @ x.
                gain = 2.0 * direction * field[index] + matrix[index, index]
                if gain >= 0 or rng.random() < math.exp(float(gain) / temperature):
                    state[index] += direction
                    field += direction * matrix[:, index]
                    energy += float(gain)
                    if energy > best_energy:
                        best_state = state.copy()
                        best_energy = energy

    return best_state


def _write_reduced_datasets(
    input_path: Path,
    headers: list[str],
    selected_indices: list[int],
    target_column: str,
    training_rows: int,
    dataset_size: int,
    train_output: TextIO,
    test_output: TextIO,
) -> None:
    columns = [headers[index] for index in selected_indices] + [target_column]
    processed_rows = 0
    # Split chunks at the exact boundary, even when it falls inside a chunk.
    for chunk in pd.read_csv(
        input_path,
        chunksize=_CSV_CHUNK_SIZE,
        dtype={target_column: str},
        encoding="utf-8-sig",
    ):
        train_count = min(len(chunk), max(training_rows - processed_rows, 0))
        if train_count:
            chunk.iloc[:train_count].loc[:, columns].to_csv(
                train_output,
                index=False,
                header=processed_rows == 0,
            )
        if train_count < len(chunk):
            chunk.iloc[train_count:].loc[:, columns].to_csv(
                test_output,
                index=False,
                header=processed_rows + train_count == training_rows,
            )
        processed_rows += len(chunk)

    if processed_rows != dataset_size:
        raise ValueError("The normalized CSV changed while writing reduced datasets")


def select_features(
    normalized_csv: str,
    reducedTrain_csv: str,
    reducedTest_csv: str,
    output_ottim_csv: str,
    output_json: str,
    target_column: str,
    percTest: float = 0.30,
    percSelected: float = 0.20,
    allowance: int = 1,
    seed: int = 42,
    alpha_computations: int = 100,
) -> dict[str, int | float | bool | list[int] | list[str]]:
    """Select features with a QUBO optimizer and write ordered train/test CSVs.

    Correlations are calculated on the first ``M = int(N * (1-percTest))``
    records only. Because the preceding preprocessing specification normalizes
    the full dataset, this selection step cannot undo that earlier exposure of
    test data to preprocessing.

    When the alpha sweep does not find a count within ``allowance`` of the
    requested count, the attempt with the closest count is used and
    ``target_reached`` is false. Ties prefer the lower alpha.
    """
    try:
        test_fraction = float(percTest)
        selected_fraction = float(percSelected)
    except (TypeError, ValueError) as exc:
        raise ValueError("percTest and percSelected must be numeric") from exc
    if not math.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError("percTest must be a finite number strictly between 0 and 1")
    if not math.isfinite(selected_fraction) or not 0 <= selected_fraction <= 1:
        raise ValueError("percSelected must be a finite number between 0 and 1")
    if isinstance(allowance, bool) or not isinstance(allowance, int) or allowance < 0:
        raise ValueError("allowance must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(alpha_computations, bool)
        or not isinstance(alpha_computations, int)
        or alpha_computations < 1
    ):
        raise ValueError("alpha_computations must be a positive integer")
    if not target_column:
        raise ValueError("target_column must not be empty")

    input_path = Path(normalized_csv)
    output_paths = [
        Path(reducedTrain_csv),
        Path(reducedTest_csv),
        Path(output_ottim_csv),
        Path(output_json),
    ]
    resolved_paths = [input_path.resolve(), *(path.resolve() for path in output_paths)]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("The input and all output files must have different paths")

    headers = _read_headers(input_path)
    if target_column not in headers:
        raise ValueError(f"Target column {target_column!r} was not found in the CSV")
    if len(headers) < 2:
        raise ValueError("The normalized CSV must contain at least one feature and the target")
    target_index = headers.index(target_column)
    feature_names = [name for index, name in enumerate(headers) if index != target_index]
    feature_count = len(feature_names)

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        raw_data_path = temporary_root / "input-data.bin"
        row_count = 0
        try:
            # Stream numeric input to disk instead of keeping the full CSV in memory.
            with raw_data_path.open("wb") as raw_file:
                chunks = pd.read_csv(
                    input_path,
                    chunksize=_CSV_CHUNK_SIZE,
                    dtype=np.float64,
                    encoding="utf-8-sig",
                )
                iterator = iter(chunks)
                while True:
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        break
                    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
                        raise ValueError(
                            f"Could not parse normalized CSV; expected numeric values: {exc}"
                        ) from exc

                    if list(chunk.columns) != headers:
                        raise ValueError("CSV columns changed while reading the normalized dataset")
                    values = chunk.to_numpy(dtype=np.float64, copy=False)
                    invalid = np.argwhere(~np.isfinite(values))
                    if invalid.size:
                        row_offset, column_index = (int(value) for value in invalid[0])
                        raise ValueError(
                            f"Row {row_count + row_offset + 2}, column "
                            f"{headers[column_index]!r} contains a missing or non-finite value"
                        )
                    target_values = values[:, target_index]
                    invalid_target = np.flatnonzero(
                        (target_values != 0.0) & (target_values != 1.0)
                    )
                    if invalid_target.size:
                        row_offset = int(invalid_target[0])
                        raise ValueError(
                            f"Row {row_count + row_offset + 2}, target column "
                            f"{target_column!r} must contain only 0 or 1"
                        )
                    values.tofile(raw_file)
                    row_count += len(chunk)
        except pd.errors.EmptyDataError as exc:
            raise ValueError("The normalized CSV is empty or has no header row") from exc

        if row_count < 2:
            raise ValueError("The normalized CSV must contain at least two data rows")
        training_rows = int(row_count * (1.0 - test_fraction))
        test_rows = row_count - training_rows
        if training_rows < 2 or test_rows < 1:
            raise ValueError(
                "The requested percTest produces an empty or unusable train/test split"
            )

        raw_data = np.memmap(
            raw_data_path,
            mode="r",
            dtype=np.float64,
            shape=(row_count, len(headers)),
        )
        matrix_start = time.perf_counter()
        target_correlations, feature_correlations = _correlation_matrices(
            raw_data,
            training_rows,
            feature_count,
            target_index,
            temporary_root / "ranked-features.bin",
            target_column,
        )
        del raw_data
        q_matrix_creation_time = time.perf_counter() - matrix_start

    target_k = int(round(selected_fraction * feature_count))
    selectable_features = np.diag(feature_correlations) > 0
    alpha_values = np.linspace(0.0, 1.0, num=alpha_computations).tolist()
    optimization_records: list[dict[str, float | int]] = []
    selections: list[np.ndarray] = []
    optimization_times: list[float] = []

    for attempt_seed, alpha in enumerate(alpha_values):
        matrix_start = time.perf_counter()
        matrix = _build_qubo(target_correlations, feature_correlations, alpha)
        q_matrix_creation_time += time.perf_counter() - matrix_start

        optimization_start = time.perf_counter()
        if alpha == 0.0:
            state = np.zeros(feature_count, dtype=np.int8)
        elif alpha == 1.0:
            state = selectable_features.astype(np.int8)
        else:
            state = _simulated_annealing(
                matrix, seed + attempt_seed, selectable_features
            )
        optimization_time = time.perf_counter() - optimization_start
        optimization_times.append(optimization_time)
        selections.append(state)
        optimization_records.append(
            {
                "alpha": float(alpha),
                "optimization_time": optimization_time,
                "n_selected": int(state.sum()),
                "objective_value": _qubo_objective(state, matrix),
            }
        )
        if abs(int(state.sum()) - target_k) <= allowance:
            break

    # If no attempt meets the tolerance, keep the closest result and report it.
    chosen_index = min(
        range(len(selections)),
        key=lambda index: (
            abs(int(selections[index].sum()) - target_k),
            alpha_values[index],
        ),
    )
    selected_vector = selections[chosen_index]
    selected_indices = [
        index for index, selected in enumerate(selected_vector) if selected
    ]
    selected_feature_names = [feature_names[index] for index in selected_indices]
    target_reached = abs(len(selected_indices) - target_k) <= allowance

    temporary_outputs: list[Path] = []
    try:
        temporary_train = _temporary_path(output_paths[0])
        temporary_outputs.append(temporary_train)
        temporary_test = _temporary_path(output_paths[1])
        temporary_outputs.append(temporary_test)
        temporary_optimization = _temporary_path(output_paths[2])
        temporary_outputs.append(temporary_optimization)
        temporary_json = _temporary_path(output_paths[3])
        temporary_outputs.append(temporary_json)

        with temporary_train.open("w", encoding="utf-8", newline="") as train_file:
            with temporary_test.open("w", encoding="utf-8", newline="") as test_file:
                _write_reduced_datasets(
                    input_path,
                    headers,
                    selected_indices,
                    target_column,
                    training_rows,
                    row_count,
                    train_file,
                    test_file,
                )

        with temporary_optimization.open(
            "w", encoding="utf-8", newline=""
        ) as optimization_file:
            writer = csv.DictWriter(
                optimization_file,
                fieldnames=[
                    "alpha",
                    "optimization_time",
                    "n_selected",
                    "objective_value",
                ],
            )
            writer.writeheader()
            writer.writerows(optimization_records)

        statistics: dict[str, int | float | bool | list[int] | list[str]] = {
            "n_features": feature_count,
            "target_ratio": selected_fraction,
            "target_k": target_k,
            "allowance": allowance,
            "n_selected": len(selected_indices),
            "alpha": float(alpha_values[chosen_index]),
            "selected_vector": [int(value) for value in selected_vector],
            "selected_feature_names": selected_feature_names,
            "algorithm": "simulated_annealing",
            "seed": seed,
            "alpha_computations": len(optimization_records),
            "percTest": test_fraction,
            "training_dataset_size": training_rows,
            "test_dataset_size": test_rows,
            "q_matrix_creation_time": q_matrix_creation_time,
            "mean_optimization_time": float(np.mean(optimization_times)),
            "std_dev_optimization_time": float(np.std(optimization_times)),
            "target_reached": target_reached,
        }
        with temporary_json.open("w", encoding="utf-8", newline="") as json_file:
            json.dump(statistics, json_file, indent=2, allow_nan=False)
            json_file.write("\n")

        for temporary_path, destination in zip(temporary_outputs, output_paths):
            os.replace(temporary_path, destination)
        return statistics
    finally:
        for temporary_path in temporary_outputs:
            if temporary_path.exists():
                temporary_path.unlink()
