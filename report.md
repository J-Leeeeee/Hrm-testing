# Fixed-Point Failure Modes and Test-Time Voting in HRM Sudoku

## Research Question

Does HRM solve Sudoku robustly, or can it converge to stable wrong fixed points?
Can Sudoku-preserving test-time transforms and voting improve accuracy at a
clear runtime cost?

## Methods

- Start from the official `sapientinc/HRM-checkpoint-sudoku-extreme` checkpoint.
- Evaluate exact accuracy, cell accuracy, invalid-board rate, clue-violation
  rate, and milliseconds per puzzle.
- Generate controlled test sets from official test labels by removing
  `1, 2, 5, 10, 20, 40` cells with a fixed seed.
- Refeed first predictions into HRM and classify outcomes as `stable_correct`,
  `stable_wrong`, `invalid_fixed_point`, `became_correct`, `became_wrong`, or
  `changed_wrong`.
- Run test-time voting with Sudoku-valid digit, row, column, and transpose
  transforms for `1, 3, 5, 10` transformed runs.

## Results

Fill this section after running the cloud experiments.

| Experiment | Key result |
| --- | --- |
| Official baseline | TBD |
| Controlled blanks | TBD |
| Fixed-point refeed | TBD |
| Test-time voting | TBD |

Expected figures:

- `figures/accuracy_by_blanks.png`
- `figures/fixed_point_failures.png`
- `figures/voting_vs_accuracy.png`
- `figures/voting_runtime.png`

## Limitations

- The MVP uses a pretrained checkpoint, so it measures behavior of the released
  Sudoku model rather than reproducing training from scratch.
- Controlled `40`-blank puzzles may be less reliably unique; report
  `unique_status`, clue violations, and validity alongside exact-match accuracy.
- Voting increases test-time compute roughly linearly with the vote count.

## Reproducibility

```bash
python -m pytest
python experiments/make_controlled_sudoku.py --limit 1000
python experiments/baseline_eval.py --checkpoint checkpoints/sudoku_extreme_hf/checkpoint --controlled-root data/controlled_sudoku --output results/controlled_baseline.csv
python experiments/fixed_point_diagnostics.py --checkpoint checkpoints/sudoku_extreme_hf/checkpoint --controlled-root data/controlled_sudoku
python experiments/test_time_voting.py --checkpoint checkpoints/sudoku_extreme_hf/checkpoint --controlled-root data/controlled_sudoku --vote-counts 1,3,5,10
python experiments/plot_results.py --baseline results/controlled_baseline.csv
```

