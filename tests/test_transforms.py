from __future__ import annotations

import numpy as np

from experiments.hrm_sudoku.sudoku import grid_to_tokens
from experiments.hrm_sudoku.transforms import make_vote_transforms, majority_vote_tokens


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


def test_vote_transforms_round_trip_tokens() -> None:
    tokens = grid_to_tokens(SOLVED)
    for transform in make_vote_transforms(12, seed=123):
        transformed = transform.apply_tokens(tokens)
        restored = transform.invert_tokens(transformed)
        np.testing.assert_array_equal(restored, tokens)


def test_majority_vote_tokens_uses_per_cell_modes() -> None:
    candidates = np.array(
        [
            [2, 3, 4],
            [2, 9, 4],
            [5, 9, 4],
        ],
        dtype=np.int16,
    )
    np.testing.assert_array_equal(majority_vote_tokens(candidates), np.array([2, 9, 4], dtype=np.int16))

