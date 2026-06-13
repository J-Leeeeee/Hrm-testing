from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import SEQ_LEN, VOCAB_SIZE


@dataclass(frozen=True)
class HRMDataset:
    inputs: np.ndarray
    labels: np.ndarray
    puzzle_identifiers: np.ndarray
    source_path: Path

    def limited(self, limit: int | None) -> "HRMDataset":
        if limit is None:
            return self
        n = min(limit, len(self.inputs))
        return HRMDataset(
            inputs=self.inputs[:n],
            labels=self.labels[:n],
            puzzle_identifiers=self.puzzle_identifiers[:n],
            source_path=self.source_path,
        )


def load_hrm_split(dataset_path: str | Path, *, split: str = "test", set_name: str = "all") -> HRMDataset:
    split_path = Path(dataset_path) / split
    inputs = np.load(split_path / f"{set_name}__inputs.npy")
    labels = np.load(split_path / f"{set_name}__labels.npy")

    puzzle_identifiers_path = split_path / f"{set_name}__puzzle_identifiers.npy"
    if puzzle_identifiers_path.exists():
        puzzle_identifiers = np.load(puzzle_identifiers_path)
    else:
        puzzle_identifiers = np.zeros((len(inputs),), dtype=np.int32)

    if inputs.ndim != 2 or inputs.shape[1] != SEQ_LEN:
        raise ValueError(f"Expected inputs shaped [N, {SEQ_LEN}], got {inputs.shape}.")
    if labels.shape != inputs.shape:
        raise ValueError(f"Expected labels shaped like inputs, got {labels.shape}.")

    return HRMDataset(
        inputs=np.asarray(inputs, dtype=np.int16),
        labels=np.asarray(labels, dtype=np.int16),
        puzzle_identifiers=np.asarray(puzzle_identifiers, dtype=np.int32),
        source_path=Path(dataset_path),
    )


def iter_batches(dataset: HRMDataset, batch_size: int) -> Iterable[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    for start in range(0, len(dataset.inputs), batch_size):
        end = min(start + batch_size, len(dataset.inputs))
        yield start, dataset.inputs[start:end], dataset.labels[start:end], dataset.puzzle_identifiers[start:end]


def write_hrm_test_dataset(
    output_dir: str | Path,
    *,
    inputs: np.ndarray,
    labels: np.ndarray,
    manifest_rows: list[dict[str, object]] | None = None,
) -> None:
    output = Path(output_dir)
    split_path = output / "test"
    split_path.mkdir(parents=True, exist_ok=True)

    inputs = np.asarray(inputs, dtype=np.int16)
    labels = np.asarray(labels, dtype=np.int16)
    if inputs.ndim != 2 or inputs.shape[1] != SEQ_LEN:
        raise ValueError(f"Expected inputs shaped [N, {SEQ_LEN}], got {inputs.shape}.")
    if labels.shape != inputs.shape:
        raise ValueError(f"Expected labels shaped like inputs, got {labels.shape}.")

    n = len(inputs)
    metadata = {
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": n,
        "mean_puzzle_examples": 1,
        "sets": ["all"],
    }

    with (split_path / "dataset.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    np.save(split_path / "all__inputs.npy", inputs)
    np.save(split_path / "all__labels.npy", labels)
    np.save(split_path / "all__puzzle_identifiers.npy", np.zeros((n,), dtype=np.int32))
    np.save(split_path / "all__puzzle_indices.npy", np.arange(n + 1, dtype=np.int32))
    np.save(split_path / "all__group_indices.npy", np.arange(n + 1, dtype=np.int32))

    with (output / "identifiers.json").open("w", encoding="utf-8") as f:
        json.dump(["<blank>"], f)

    if manifest_rows is not None:
        with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = sorted({key for row in manifest_rows for key in row})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

