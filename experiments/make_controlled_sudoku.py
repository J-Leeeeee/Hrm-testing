from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hrm_sudoku.controlled import make_puzzle
from hrm_sudoku.datasets import load_hrm_split, write_hrm_test_dataset
from hrm_sudoku.sudoku import count_solutions, grid_to_tokens, tokens_to_grid


def parse_blank_counts(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build controlled near-complete Sudoku test sets.")
    parser.add_argument("--data-path", default="data/sudoku-extreme-1k-aug-1000")
    parser.add_argument("--output-dir", default="data/controlled_sudoku")
    parser.add_argument("--blank-counts", default="1,2,5,10,20,40")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=50)
    parser.add_argument("--skip-uniqueness", action="store_true", help="Skip the capped backtracking uniqueness check.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_hrm_split(args.data_path, split="test").limited(args.limit)
    solutions = [tokens_to_grid(label) for label in dataset.labels]
    blank_counts = parse_blank_counts(args.blank_counts)
    root = Path(args.output_dir)

    for blank_count in blank_counts:
        rng = np.random.default_rng(args.seed + blank_count)
        inputs = []
        labels = []
        manifest_rows: list[dict[str, object]] = []

        for source_index, solution in enumerate(solutions):
            status = "unknown"
            attempts = 0
            puzzle = None
            for attempts in range(1, args.max_attempts + 1):
                candidate = make_puzzle(solution, blank_count, rng)
                if args.skip_uniqueness:
                    puzzle = candidate
                    break

                solution_count = count_solutions(candidate, max_solutions=2)
                status = "unique" if solution_count == 1 else "ambiguous" if solution_count > 1 else "unsolved"
                puzzle = candidate
                if status == "unique":
                    break

            assert puzzle is not None
            inputs.append(grid_to_tokens(puzzle))
            labels.append(grid_to_tokens(solution))
            manifest_rows.append(
                {
                    "example_id": source_index,
                    "source_index": source_index,
                    "blank_count": blank_count,
                    "attempts": attempts,
                    "unique_status": status,
                    "seed": args.seed,
                }
            )

        output = root / f"blanks_{blank_count}"
        write_hrm_test_dataset(output, inputs=np.stack(inputs), labels=np.stack(labels), manifest_rows=manifest_rows)
        unique = sum(1 for row in manifest_rows if row["unique_status"] == "unique")
        print(f"Wrote {len(inputs)} puzzles to {output} ({unique} unique by capped check).")


if __name__ == "__main__":
    main()
