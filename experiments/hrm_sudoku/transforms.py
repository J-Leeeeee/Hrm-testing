from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sudoku import board_metrics, grid_to_tokens, tokens_to_grid, unit_conflicts, clue_violations, is_complete_solution


@dataclass(frozen=True)
class SudokuTransform:
    digit_map: np.ndarray
    inverse_digit_map: np.ndarray
    transpose: bool
    row_perm: np.ndarray
    col_perm: np.ndarray

    def apply_grid(self, grid: np.ndarray) -> np.ndarray:
        work = np.asarray(grid)
        if self.transpose:
            work = work.T
        return self.digit_map[work[self.row_perm, :][:, self.col_perm]]

    def invert_grid(self, transformed_grid: np.ndarray) -> np.ndarray:
        values = self.inverse_digit_map[np.asarray(transformed_grid)]
        unpermuted = np.zeros_like(values)
        for new_row, old_row in enumerate(self.row_perm):
            for new_col, old_col in enumerate(self.col_perm):
                unpermuted[old_row, old_col] = values[new_row, new_col]
        if self.transpose:
            unpermuted = unpermuted.T
        return unpermuted

    def apply_tokens(self, tokens: np.ndarray) -> np.ndarray:
        return grid_to_tokens(self.apply_grid(tokens_to_grid(tokens, allow_pad=True)))

    def invert_tokens(self, tokens: np.ndarray) -> np.ndarray:
        return grid_to_tokens(self.invert_grid(tokens_to_grid(tokens, allow_pad=True)))


def identity_transform() -> SudokuTransform:
    digits = np.arange(10, dtype=np.int8)
    return SudokuTransform(
        digit_map=digits,
        inverse_digit_map=digits,
        transpose=False,
        row_perm=np.arange(9, dtype=np.int8),
        col_perm=np.arange(9, dtype=np.int8),
    )


def random_transform(rng: np.random.Generator) -> SudokuTransform:
    digit_map = np.arange(10, dtype=np.int8)
    digit_map[1:] = rng.permutation(np.arange(1, 10, dtype=np.int8))
    inverse_digit_map = np.zeros_like(digit_map)
    for source_digit, mapped_digit in enumerate(digit_map):
        inverse_digit_map[mapped_digit] = source_digit

    bands = rng.permutation(3)
    row_perm = np.concatenate([band * 3 + rng.permutation(3) for band in bands]).astype(np.int8)
    stacks = rng.permutation(3)
    col_perm = np.concatenate([stack * 3 + rng.permutation(3) for stack in stacks]).astype(np.int8)

    return SudokuTransform(
        digit_map=digit_map,
        inverse_digit_map=inverse_digit_map,
        transpose=bool(rng.random() < 0.5),
        row_perm=row_perm,
        col_perm=col_perm,
    )


def make_vote_transforms(votes: int, *, seed: int) -> list[SudokuTransform]:
    if votes < 1:
        raise ValueError("votes must be at least 1.")
    rng = np.random.default_rng(seed)
    return [identity_transform()] + [random_transform(rng) for _ in range(votes - 1)]


def majority_vote_tokens(candidates: np.ndarray) -> np.ndarray:
    arr = np.asarray(candidates, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"Expected candidates shaped [votes, cells], got {arr.shape}.")
    out = []
    for col in range(arr.shape[1]):
        values, counts = np.unique(arr[:, col], return_counts=True)
        out.append(int(values[np.argmax(counts)]))
    return np.asarray(out, dtype=np.int16)


def select_best_candidate(input_tokens: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Choose the most Sudoku-plausible candidate without using the label."""
    input_grid = tokens_to_grid(input_tokens, allow_pad=True)

    best_idx = 0
    best_score: tuple[int, int, int] | None = None
    for idx, candidate in enumerate(np.asarray(candidates)):
        grid = tokens_to_grid(candidate, allow_pad=True)
        score = (
            0 if is_complete_solution(grid) else 1,
            clue_violations(input_grid, grid),
            unit_conflicts(grid),
        )
        if best_score is None or score < best_score:
            best_idx = idx
            best_score = score
    return np.asarray(candidates[best_idx], dtype=np.int16)

