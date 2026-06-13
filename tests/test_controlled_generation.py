from __future__ import annotations

import numpy as np

from experiments.hrm_sudoku.sudoku import grid_to_tokens, tokens_to_grid
from experiments.hrm_sudoku.controlled import make_puzzle


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


def test_controlled_generator_blanks_exactly_k_cells_and_preserves_label() -> None:
    puzzle = make_puzzle(SOLVED, 5, np.random.default_rng(0))
    assert int(np.sum(puzzle == 0)) == 5

    input_tokens = grid_to_tokens(puzzle)
    label_tokens = grid_to_tokens(SOLVED)

    assert int(np.sum(tokens_to_grid(input_tokens) == 0)) == 5
    np.testing.assert_array_equal(tokens_to_grid(label_tokens), SOLVED)
