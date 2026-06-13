from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sudoku import board_metrics


@dataclass(frozen=True)
class FixedPointOutcome:
    outcome: str
    changed_after_refeed: bool
    first_exact: bool
    second_exact: bool
    first_invalid: bool
    second_invalid: bool


def classify_fixed_point(input_tokens: np.ndarray, label_tokens: np.ndarray, first_pred: np.ndarray, second_pred: np.ndarray) -> FixedPointOutcome:
    changed = not np.array_equal(first_pred, second_pred)
    first = board_metrics(input_tokens, label_tokens, first_pred)
    second = board_metrics(first_pred, label_tokens, second_pred)

    if not changed and first.exact:
        outcome = "stable_correct"
    elif not changed:
        outcome = "stable_wrong"
    elif (not first.exact) and second.exact:
        outcome = "became_correct"
    elif first.exact and (not second.exact):
        outcome = "became_wrong"
    elif first.exact and second.exact:
        outcome = "changed_still_correct"
    else:
        outcome = "changed_wrong"

    if outcome == "stable_wrong" and first.invalid:
        outcome = "invalid_fixed_point"

    return FixedPointOutcome(
        outcome=outcome,
        changed_after_refeed=changed,
        first_exact=first.exact,
        second_exact=second.exact,
        first_invalid=first.invalid,
        second_invalid=second.invalid,
    )

