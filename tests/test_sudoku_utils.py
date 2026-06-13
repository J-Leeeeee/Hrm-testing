from __future__ import annotations

import numpy as np

from experiments.hrm_sudoku.sudoku import board_metrics, count_solutions, grid_to_tokens, tokens_to_grid


SOLVED = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [4, 5, 6, 7, 8, 9, 1, 2, 3],
        [7, 8, 9, 1, 2, 3, 4, 5, 6],
        [2, 3, 4, 5, 6, 7, 8, 9, 1],
        [5, 6, 7, 8, 9, 1, 2, 3, 4],
        [8, 9, 1, 2, 3, 4, 5, 6, 7],
        [3, 4, 5, 6, 7, 8, 9, 1, 2],
        [6, 7, 8, 9, 1, 2, 3, 4, 5],
        [9, 1, 2, 3, 4, 5, 6, 7, 8],
    ],
    dtype=np.int8,
)


def test_token_round_trip_preserves_blanks_and_digits() -> None:
    puzzle = SOLVED.copy()
    puzzle[0, 0] = 0

    tokens = grid_to_tokens(puzzle)

    assert tokens[0] == 1
    assert tokens[1] == 3
    np.testing.assert_array_equal(tokens_to_grid(tokens), puzzle)


def test_validator_catches_row_column_box_and_clue_violations() -> None:
    label = grid_to_tokens(SOLVED)
    puzzle = SOLVED.copy()
    puzzle[0, 0] = 0
    puzzle_tokens = grid_to_tokens(puzzle)

    row_bad = SOLVED.copy()
    row_bad[0, 0] = row_bad[0, 1]
    assert board_metrics(puzzle_tokens, label, grid_to_tokens(row_bad)).invalid

    col_bad = SOLVED.copy()
    col_bad[0, 0] = col_bad[1, 0]
    assert board_metrics(puzzle_tokens, label, grid_to_tokens(col_bad)).invalid

    box_bad = SOLVED.copy()
    box_bad[0, 0] = box_bad[1, 1]
    assert board_metrics(puzzle_tokens, label, grid_to_tokens(box_bad)).invalid

    clue_bad = SOLVED.copy()
    clue_bad[0, 1] = 9
    metrics = board_metrics(puzzle_tokens, label, grid_to_tokens(clue_bad))
    assert metrics.invalid
    assert metrics.clue_violations == 1


def test_solution_counter_distinguishes_solved_and_invalid_puzzles() -> None:
    assert count_solutions(SOLVED, max_solutions=2) == 1

    invalid = SOLVED.copy()
    invalid[0, 0] = invalid[0, 1]
    assert count_solutions(invalid, max_solutions=2) == 0

