from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .constants import BLANK_TOKEN, DIGIT_TOKEN_OFFSET, GRID_SIZE, SEQ_LEN


def tokens_to_grid(tokens: np.ndarray, *, allow_pad: bool = False) -> np.ndarray:
    """Convert HRM Sudoku tokens to 0..9 grid values.

    HRM stores Sudoku as PAD=0, blank=1, and digit d as d+1. PAD is not a
    Sudoku value, so callers must opt into treating PAD as blank.
    """
    arr = np.asarray(tokens, dtype=np.int64)
    if arr.size != SEQ_LEN:
        raise ValueError(f"Expected {SEQ_LEN} tokens, got {arr.size}.")

    flat = arr.reshape(-1).copy()
    if allow_pad:
        flat[flat == 0] = BLANK_TOKEN

    if np.any((flat < BLANK_TOKEN) | (flat > 10)):
        bad = sorted(set(flat[(flat < BLANK_TOKEN) | (flat > 10)].tolist()))
        raise ValueError(f"Unexpected Sudoku token(s): {bad}")

    return (flat - DIGIT_TOKEN_OFFSET).reshape(GRID_SIZE, GRID_SIZE).astype(np.int8)


def grid_to_tokens(grid: np.ndarray) -> np.ndarray:
    """Convert a 0..9 Sudoku grid to HRM tokens."""
    arr = np.asarray(grid, dtype=np.int64)
    if arr.size != SEQ_LEN:
        raise ValueError(f"Expected {SEQ_LEN} cells, got {arr.size}.")

    flat = arr.reshape(-1)
    if np.any((flat < 0) | (flat > 9)):
        bad = sorted(set(flat[(flat < 0) | (flat > 9)].tolist()))
        raise ValueError(f"Unexpected Sudoku cell value(s): {bad}")

    return (flat + DIGIT_TOKEN_OFFSET).astype(np.int16)


def prediction_to_input_tokens(pred_tokens: np.ndarray) -> np.ndarray:
    """Normalize model predictions before refeeding them as a new puzzle input.

    The model can technically emit PAD=0. PAD is not a Sudoku clue token, so
    fixed-point refeeding treats PAD and blank predictions as blank cells.
    """
    arr = np.asarray(pred_tokens, dtype=np.int64).reshape(-1).copy()
    arr[arr <= BLANK_TOKEN] = BLANK_TOKEN
    arr[arr > 10] = BLANK_TOKEN
    return arr.astype(np.int16)


def _units(board: np.ndarray) -> Iterable[np.ndarray]:
    for row in range(GRID_SIZE):
        yield board[row, :]
    for col in range(GRID_SIZE):
        yield board[:, col]
    for box_row in range(0, GRID_SIZE, 3):
        for box_col in range(0, GRID_SIZE, 3):
            yield board[box_row:box_row + 3, box_col:box_col + 3].reshape(-1)


def unit_conflicts(board: np.ndarray) -> int:
    """Count repeated nonzero digits across all rows, columns, and boxes."""
    conflicts = 0
    for unit in _units(np.asarray(board)):
        values = [int(v) for v in unit if int(v) != 0]
        conflicts += len(values) - len(set(values))
    return conflicts


def is_complete_solution(board: np.ndarray) -> bool:
    arr = np.asarray(board)
    if arr.shape != (GRID_SIZE, GRID_SIZE):
        return False
    if np.any((arr < 1) | (arr > 9)):
        return False
    return unit_conflicts(arr) == 0


def clue_violations(input_grid: np.ndarray, pred_grid: np.ndarray) -> int:
    clues = np.asarray(input_grid) != 0
    return int(np.sum(np.asarray(pred_grid)[clues] != np.asarray(input_grid)[clues]))


@dataclass(frozen=True)
class BoardMetrics:
    exact: bool
    cell_accuracy: float
    invalid: bool
    complete_valid: bool
    clue_violations: int
    unit_conflicts: int


def board_metrics(input_tokens: np.ndarray, label_tokens: np.ndarray, pred_tokens: np.ndarray) -> BoardMetrics:
    input_grid = tokens_to_grid(input_tokens, allow_pad=True)
    label_grid = tokens_to_grid(label_tokens, allow_pad=True)
    pred_grid = tokens_to_grid(pred_tokens, allow_pad=True)

    exact = bool(np.array_equal(pred_grid, label_grid))
    complete_valid = is_complete_solution(pred_grid)
    violations = clue_violations(input_grid, pred_grid)
    conflicts = unit_conflicts(pred_grid)
    invalid = (not complete_valid) or (violations > 0)
    cell_accuracy = float(np.mean(pred_grid == label_grid))

    return BoardMetrics(
        exact=exact,
        cell_accuracy=cell_accuracy,
        invalid=invalid,
        complete_valid=complete_valid,
        clue_violations=violations,
        unit_conflicts=conflicts,
    )


def count_solutions(puzzle: np.ndarray, *, max_solutions: int = 2) -> int:
    """Count Sudoku solutions with a small MRV backtracker.

    Stops once max_solutions is reached, which is enough to distinguish unique
    from ambiguous generated controlled puzzles.
    """
    board = np.asarray(puzzle, dtype=np.int8).copy()
    if board.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"Expected a 9x9 puzzle, got {board.shape}.")
    if unit_conflicts(board) > 0:
        return 0

    row_used = [set(int(v) for v in board[r, :] if int(v) != 0) for r in range(GRID_SIZE)]
    col_used = [set(int(v) for v in board[:, c] if int(v) != 0) for c in range(GRID_SIZE)]
    box_used = [
        set(int(v) for v in board[br:br + 3, bc:bc + 3].reshape(-1) if int(v) != 0)
        for br in range(0, GRID_SIZE, 3)
        for bc in range(0, GRID_SIZE, 3)
    ]

    def box_index(row: int, col: int) -> int:
        return (row // 3) * 3 + (col // 3)

    def candidates(row: int, col: int) -> list[int]:
        used = row_used[row] | col_used[col] | box_used[box_index(row, col)]
        return [digit for digit in range(1, 10) if digit not in used]

    solutions = 0

    def search() -> None:
        nonlocal solutions
        if solutions >= max_solutions:
            return

        best_cell: tuple[int, int] | None = None
        best_candidates: list[int] | None = None
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if board[row, col] != 0:
                    continue
                cell_candidates = candidates(row, col)
                if not cell_candidates:
                    return
                if best_candidates is None or len(cell_candidates) < len(best_candidates):
                    best_cell = (row, col)
                    best_candidates = cell_candidates

        if best_cell is None:
            solutions += 1
            return

        row, col = best_cell
        assert best_candidates is not None
        box = box_index(row, col)
        for digit in best_candidates:
            board[row, col] = digit
            row_used[row].add(digit)
            col_used[col].add(digit)
            box_used[box].add(digit)

            search()

            row_used[row].remove(digit)
            col_used[col].remove(digit)
            box_used[box].remove(digit)
            board[row, col] = 0

            if solutions >= max_solutions:
                return

    search()
    return solutions

