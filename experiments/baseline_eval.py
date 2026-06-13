from __future__ import annotations

import argparse
import time
from pathlib import Path

from hrm_sudoku.datasets import HRMDataset, iter_batches, load_hrm_split
from hrm_sudoku.model import load_runner
from hrm_sudoku.reporting import print_summary, write_csv
from hrm_sudoku.sudoku import board_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate HRM Sudoku exact/cell/validity metrics.")
    parser.add_argument("--checkpoint", default="checkpoints/sudoku_extreme_hf/checkpoint")
    parser.add_argument("--data-path", default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--controlled-root", default=None, help="Optional root containing blanks_* HRM test datasets.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/baseline.csv")
    return parser.parse_args()


def dataset_specs(args: argparse.Namespace) -> list[tuple[str, int | None, HRMDataset]]:
    if args.controlled_root is None:
        return [("official", None, load_hrm_split(args.data_path, split=args.split).limited(args.limit))]

    root = Path(args.controlled_root)
    specs = []
    for path in sorted(root.glob("blanks_*"), key=lambda p: int(p.name.split("_")[-1])):
        blank_count = int(path.name.split("_")[-1])
        specs.append((path.name, blank_count, load_hrm_split(path, split="test").limited(args.limit)))
    if not specs:
        raise FileNotFoundError(f"No blanks_* datasets found under {root}.")
    return specs


def main() -> None:
    args = parse_args()
    runner = load_runner(checkpoint=args.checkpoint, data_path=args.data_path, batch_size=args.batch_size)

    rows: list[dict[str, object]] = []
    for dataset_name, blank_count, dataset in dataset_specs(args):
        for start, inputs, labels, puzzle_ids in iter_batches(dataset, args.batch_size):
            runner.synchronize()
            t0 = time.perf_counter()
            preds = runner.predict(inputs, labels, puzzle_ids)
            runner.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_example_ms = elapsed_ms / max(len(inputs), 1)

            for offset, (inp, label, pred) in enumerate(zip(inputs, labels, preds)):
                metrics = board_metrics(inp, label, pred)
                rows.append(
                    {
                        "experiment": "baseline",
                        "dataset": dataset_name,
                        "blank_count": "" if blank_count is None else blank_count,
                        "example_id": start + offset,
                        "exact": int(metrics.exact),
                        "cell_accuracy": metrics.cell_accuracy,
                        "invalid": int(metrics.invalid),
                        "complete_valid": int(metrics.complete_valid),
                        "clue_violations": metrics.clue_violations,
                        "unit_conflicts": metrics.unit_conflicts,
                        "elapsed_ms": per_example_ms,
                        "batch_size": args.batch_size,
                        "checkpoint": str(runner.checkpoint_file),
                    }
                )

    write_csv(args.output, rows)
    print_summary(rows, group_key="dataset")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

