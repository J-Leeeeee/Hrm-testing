# Fixed-Point Failure Modes and Test-Time Voting in HRM Sudoku

## Research Question

Does HRM solve Sudoku robustly, or can it converge to stable wrong fixed points?
Can Sudoku-preserving test-time transforms and voting improve accuracy at a
clear runtime cost?

## Methods

- Start from the official `sapientinc/HRM-checkpoint-sudoku-extreme` checkpoint.
- Evaluate exact accuracy, cell accuracy, invalid-board rate, clue-violation
  rate, and milliseconds per puzzle.
- Generate controlled test sets from 1,000 official test labels by removing
  `1, 2, 5, 10, 20, 40` cells with a fixed seed. A capped backtracking check
  classified all 6,000 generated boards as uniquely solvable.
- Refeed first predictions into HRM and classify outcomes as `stable_correct`,
  `stable_wrong`, `invalid_fixed_point`, `became_correct`, `became_wrong`, or
  `changed_wrong`.
- In a preliminary experiment, run three-way test-time voting with
  Sudoku-preserving digit, row, column, and transpose transforms on 16 puzzles
  per missing-cell count.

## Results

The one-pass baseline and fixed-point refeed experiments each used **1,000
puzzles per missing-cell count** (**6,000 puzzles total**). The preliminary
voting experiment used **16 puzzles per missing-cell count** (**96 puzzles**) and
three transformed runs per puzzle. Sample sizes are kept explicit because the
voting estimate is much less precise than the main results.

| Experiment | Key result |
| --- | --- |
| Controlled one-pass evaluation | Pooled exact-match accuracy was **62.0%**, cell accuracy was **93.4%**, and the invalid-output rate was **38.1%** across 6,000 boards. |
| Missing-cell perturbation | Exact-match accuracy rose from **9.0%** (1 missing cell) through **16.6%**, **56.7%**, **92.5%**, and **96.9%** to **100.0%** (40 missing cells). Invalid outputs fell from **91.0%** to **0.0%** over the same range. |
| Fixed-point refeed | Of 3,717 initially exact boards, **3,486 (93.8%)** became wrong after one refeed. Of 2,283 initial errors, only **14 (0.6%)** became correct; **168 (7.4%)** were unchanged invalid fixed points. Across all boards, outcomes were 58.1% `became_wrong`, 35.0% `changed_wrong`, 3.8% `stable_correct`, 2.8% `invalid_fixed_point`, and 0.2% `became_correct`. |
| Preliminary test-time voting | On 96 puzzles with three transformed runs, exact match was **62.5%** for the single prediction, **70.8%** for majority voting, and **76.0%** for validity-aware reranking. The shared three-run inference averaged **427 ms/puzzle**, compared with **16.6 ms/puzzle** for the earlier 32-puzzle batched pilot baseline; this observed 26× wall-clock ratio combines extra test-time inference with different batching strategies. |

Figures:

- `figures/accuracy_by_blanks.png`
- `figures/fixed_point_failures.png`
- `figures/voting_vs_accuracy.png`

## Limitations

- The study uses a pretrained checkpoint, so it measures behavior of the released
  Sudoku model rather than reproducing training from scratch.
- The controlled inputs are generated from official test labels and isolate
  missing-cell count; they are not a representative sample of human-rated
  Sudoku difficulty. The reversed accuracy pattern likely reflects distribution
  sensitivity and does not imply that puzzles become intrinsically easier as
  clues are removed.
- The uniqueness check is capped at 50 attempts per source board. All generated
  boards used here were classified as unique within that procedure.
- The voting result is preliminary (`n=96`) and should not be presented as if it
  had the same precision as the 6,000-board baseline and refeed studies.
- The reported voting wall-clock ratio compares per-puzzle transformed inference
  with a separately batched baseline, so it is an observed system-level cost,
  not an isolated estimate of transformation overhead.

## Reproducibility

```bash
python -m pytest
python experiments/make_controlled_sudoku.py --limit 1000
python experiments/baseline_eval.py --checkpoint checkpoints/sudoku_extreme_hf/checkpoint --controlled-root data/controlled_sudoku --output results/controlled_baseline.csv
python experiments/fixed_point_diagnostics.py --checkpoint checkpoints/sudoku_extreme_hf/checkpoint --controlled-root data/controlled_sudoku
python experiments/test_time_voting.py --checkpoint checkpoints/sudoku_extreme_hf/checkpoint --controlled-root data/controlled_sudoku --votes 3 --limit 16 --output results/voting_n16_v3.csv
python experiments/plot_results.py --baseline results/controlled_baseline.csv --fixed-point results/fixed_point.csv --voting results/voting_n16_v3.csv
```

