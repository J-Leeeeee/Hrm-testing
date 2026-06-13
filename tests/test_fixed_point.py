from __future__ import annotations

import numpy as np

from experiments.hrm_sudoku.fixed_point import classify_fixed_point
from experiments.hrm_sudoku.sudoku import grid_to_tokens


SOLVED_A = np.array(
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

SOLVED_B = ((SOLVED_A % 9) + 1).astype(np.int8)


def test_classify_fixed_point_stable_wrong() -> None:
    blank_input = grid_to_tokens(np.zeros((9, 9), dtype=np.int8))
    label = grid_to_tokens(SOLVED_A)
    wrong = grid_to_tokens(SOLVED_B)

    outcome = classify_fixed_point(blank_input, label, wrong, wrong)

    assert outcome.outcome == "stable_wrong"
    assert not outcome.changed_after_refeed
    assert not outcome.first_exact


def test_classify_fixed_point_became_correct() -> None:
    blank_input = grid_to_tokens(np.zeros((9, 9), dtype=np.int8))
    label = grid_to_tokens(SOLVED_A)
    wrong = grid_to_tokens(SOLVED_B)

    outcome = classify_fixed_point(blank_input, label, wrong, label)

    assert outcome.outcome == "became_correct"
    assert outcome.changed_after_refeed
    assert outcome.second_exact

