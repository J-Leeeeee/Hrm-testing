from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from hrm_sudoku.datasets import HRMDataset, load_hrm_split
from hrm_sudoku.model import load_runner
from hrm_sudoku.reporting import print_summary, write_csv
from hrm_sudoku.sudoku import board_metrics
from hrm_sudoku.transforms import make_vote_transforms, majority_vote_tokens, select_best_candidate


def parse_vote_counts(args: argparse.Namespace) -> list[int]:
    if args.votes is not None:
        return [args.votes]
    return [int(part.strip()) for part in args.vote_counts.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HRM Sudoku test-time transform voting.")
    parser.add_argument("--checkpoint", default="checkpoints/sudoku_extreme_hf/checkpoint")
    parser.add_argument("--data-path", default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--controlled-root", default=None)
    parser.add_argument("--votes", type=int, default=None, help="Run one vote count, e.g. --votes 3.")
    parser.add_argument("--vote-counts", default="1,3,5,10")
    parser.add_argument("--batch-size", type=int, default=None, help="Defaults to the largest requested vote count.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="results/voting.csv")
    return parser.parse_args()


def dataset_specs(args: argparse.Namespace) -> list[tuple[str, int | None, HRMDataset]]:
    if args.controlled_root is None:
        return [("official", None, load_hrm_split(args.data_path, split="test").limited(args.limit))]

    root = Path(args.controlled_root)
    specs = []
    for path in sorted(root.glob("blanks_*"), key=lambda p: int(p.name.split("_")[-1])):
        blank_count = int(path.name.split("_")[-1])
        specs.append((path.name, blank_count, load_hrm_split(path, split="test").limited(args.limit)))
    if not specs:
        raise FileNotFoundError(f"No blanks_* datasets found under {root}.")
    return specs


def run_one_puzzle(runner, inp: np.ndarray, label: np.ndarray, puzzle_id: int, votes: int, seed: int) -> tuple[dict[str, np.ndarray], float]:
    transforms = make_vote_transforms(votes, seed=seed)
    transformed_inputs = np.stack([transform.apply_tokens(inp) for transform in transforms])
    transformed_labels = np.stack([transform.apply_tokens(label) for transform in transforms])
    puzzle_ids = np.full((votes,), puzzle_id, dtype=np.int32)

    runner.synchronize()
    t0 = time.perf_counter()
    transformed_preds = runner.predict(transformed_inputs, transformed_labels, puzzle_ids)
    runner.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    candidates = np.stack([transform.invert_tokens(pred) for transform, pred in zip(transforms, transformed_preds)])
    predictions = {
        "single": candidates[0],
        "majority": majority_vote_tokens(candidates),
        "rerank": select_best_candidate(inp, candidates),
    }
    return predictions, elapsed_ms


def main() -> None:
    args = parse_args()
    vote_counts = parse_vote_counts(args)
    max_votes = max(vote_counts)
    runner_batch_size = args.batch_size if args.batch_size is not None else max_votes
    runner = load_runner(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        batch_size=max(runner_batch_size, max_votes),
    )

    rows: list[dict[str, object]] = []
    for dataset_name, blank_count, dataset in dataset_specs(args):
        for vote_count in vote_counts:
            for example_id, (inp, label, puzzle_id) in enumerate(zip(dataset.inputs, dataset.labels, dataset.puzzle_identifiers)):
                predictions, elapsed_ms = run_one_puzzle(
                    runner,
                    inp,
                    label,
                    int(puzzle_id),
                    vote_count,
                    seed=args.seed + example_id * 9973 + vote_count,
                )

                for method, pred in predictions.items():
                    metrics = board_metrics(inp, label, pred)
                    rows.append(
                        {
                            "experiment": "test_time_voting",
                            "dataset": dataset_name,
                            "blank_count": "" if blank_count is None else blank_count,
                            "example_id": example_id,
                            "method": method,
                            "votes": vote_count,
                            "exact": int(metrics.exact),
                            "cell_accuracy": metrics.cell_accuracy,
                            "invalid": int(metrics.invalid),
                            "complete_valid": int(metrics.complete_valid),
                            "clue_violations": metrics.clue_violations,
                            "unit_conflicts": metrics.unit_conflicts,
                            "elapsed_ms": elapsed_ms,
                            "runtime_multiplier": vote_count,
                            "batch_size": runner.batch_size,
                            "checkpoint": str(runner.checkpoint_file),
                            "seed": args.seed,
                        }
                    )

    write_csv(args.output, rows)
    print_summary(rows, group_key="method")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
