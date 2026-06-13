from __future__ import annotations

import numpy as np


def make_puzzle(solution_grid: np.ndarray, blank_count: int, rng: np.random.Generator) -> np.ndarray:
    puzzle = np.asarray(solution_grid).copy()
    blank_positions = rng.choice(81, size=blank_count, replace=False)
    puzzle.reshape(-1)[blank_positions] = 0
    return puzzle

