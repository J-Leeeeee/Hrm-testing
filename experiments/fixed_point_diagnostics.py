from __future__ import annotations

import argparse
import time
from pathlib import Path

from hrm_sudoku.datasets import HRMDataset, iter_batches, load_hrm_split
from hrm_sudoku.fixed_point import classify_fixed_point
from hrm_sudoku.model import load_runner
from hrm_sudoku.reporting import write_csv
from hrm_sudoku.sudoku import board_metrics, prediction_to_input_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refeed HRM Sudoku predictions and classify fixed-point outcomes.")
    parser.add_argument("--checkpoint", default="checkpoints/sudoku_extreme_hf/checkpoint")
    parser.add_argument("--data-path", default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--controlled-root", default=None)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/fixed_point.csv")
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


def main() -> None:
    args = parse_args()
    runner = load_runner(checkpoint=args.checkpoint, data_path=args.data_path, batch_size=args.batch_size)

    rows: list[dict[str, object]] = []
    for dataset_name, blank_count, dataset in dataset_specs(args):
        for start, inputs, labels, puzzle_ids in iter_batches(dataset, args.batch_size):
            runner.synchronize()
            t0 = time.perf_counter()
            first_preds = runner.predict(inputs, labels, puzzle_ids)
            refeed_inputs = [prediction_to_input_tokens(pred) for pred in first_preds]
            second_preds = runner.predict(refeed_inputs, labels, puzzle_ids)
            runner.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(len(inputs), 1)

            for offset, (inp, label, first, second) in enumerate(zip(inputs, labels, first_preds, second_preds)):
                outcome = classify_fixed_point(inp, label, first, second)
                first_metrics = board_metrics(inp, label, first)
                second_metrics = board_metrics(first, label, second)
                rows.append(
                    {
                        "experiment": "fixed_point",
                        "dataset": dataset_name,
                        "blank_count": "" if blank_count is None else blank_count,
                        "example_id": start + offset,
                        "outcome": outcome.outcome,
                        "changed_after_refeed": int(outcome.changed_after_refeed),
                        "exact": int(first_metrics.exact),
                        "second_exact": int(second_metrics.exact),
                        "cell_accuracy": first_metrics.cell_accuracy,
                        "second_cell_accuracy": second_metrics.cell_accuracy,
                        "invalid": int(first_metrics.invalid),
                        "second_invalid": int(second_metrics.invalid),
                        "complete_valid": int(first_metrics.complete_valid),
                        "second_complete_valid": int(second_metrics.complete_valid),
                        "clue_violations": first_metrics.clue_violations,
                        "second_clue_violations": second_metrics.clue_violations,
                        "unit_conflicts": first_metrics.unit_conflicts,
                        "second_unit_conflicts": second_metrics.unit_conflicts,
                        "elapsed_ms": elapsed_ms,
                        "batch_size": args.batch_size,
                        "checkpoint": str(runner.checkpoint_file),
                    }
                )

    write_csv(args.output, rows)
    by_outcome = {}
    for row in rows:
        by_outcome[row["outcome"]] = by_outcome.get(row["outcome"], 0) + 1
    print(by_outcome)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

