from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot HRM Sudoku fixed-point and voting study results.")
    parser.add_argument("--baseline", default="results/baseline.csv")
    parser.add_argument("--fixed-point", default="results/fixed_point.csv")
    parser.add_argument("--voting", default="results/voting.csv")
    parser.add_argument("--output-dir", default="figures")
    return parser.parse_args()


def has_data(path: str | Path) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size > 0


def save_accuracy_by_blanks(baseline_path: str | Path, output_dir: Path) -> None:
    if not has_data(baseline_path):
        return
    df = pd.read_csv(baseline_path)
    if "blank_count" not in df.columns:
        return
    df = df[df["blank_count"].notna()]
    if df.empty:
        return
    df["blank_count"] = df["blank_count"].astype(int)
    summary = df.groupby("blank_count", as_index=False).agg(
        exact_accuracy=("exact", "mean"),
        cell_accuracy=("cell_accuracy", "mean"),
        invalid_rate=("invalid", "mean"),
    )

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=summary, x="blank_count", y="exact_accuracy", marker="o", label="Exact")
    sns.lineplot(data=summary, x="blank_count", y="cell_accuracy", marker="o", label="Cell")
    sns.lineplot(data=summary, x="blank_count", y="invalid_rate", marker="o", label="Invalid")
    plt.xlabel("Missing cells")
    plt.ylabel("Rate")
    plt.ylim(-0.02, 1.02)
    plt.title("HRM accuracy by controlled blank count")
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_blanks.png", dpi=200)
    plt.close()


def save_fixed_point_failures(fixed_point_path: str | Path, output_dir: Path) -> None:
    if not has_data(fixed_point_path):
        return
    df = pd.read_csv(fixed_point_path)
    if "blank_count" not in df.columns:
        return
    df = df[df["blank_count"].notna()]
    if df.empty:
        return
    df["blank_count"] = df["blank_count"].astype(int)
    summary = df.groupby(["blank_count", "outcome"]).size().reset_index(name="count")
    totals = summary.groupby("blank_count")["count"].transform("sum")
    summary["rate"] = summary["count"] / totals

    plt.figure(figsize=(8, 4.5))
    sns.barplot(data=summary, x="blank_count", y="rate", hue="outcome")
    plt.xlabel("Missing cells")
    plt.ylabel("Outcome rate")
    plt.ylim(0, 1)
    plt.title("Fixed-point outcomes after refeeding HRM predictions")
    plt.legend(title="Outcome", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "fixed_point_failures.png", dpi=200)
    plt.close()


def save_voting_tradeoff(voting_path: str | Path, output_dir: Path) -> None:
    if not has_data(voting_path):
        return
    df = pd.read_csv(voting_path)
    if df.empty:
        return
    summary = df.groupby(["votes", "method"], as_index=False).agg(
        exact_accuracy=("exact", "mean"),
        invalid_rate=("invalid", "mean"),
        elapsed_ms=("elapsed_ms", "mean"),
    )

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=summary, x="votes", y="exact_accuracy", hue="method", marker="o")
    plt.xlabel("Test-time transformed runs")
    plt.ylabel("Exact accuracy")
    plt.ylim(-0.02, 1.02)
    plt.title("Voting accuracy vs test-time compute")
    plt.tight_layout()
    plt.savefig(output_dir / "voting_vs_accuracy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.lineplot(data=summary, x="votes", y="elapsed_ms", hue="method", marker="o")
    plt.xlabel("Test-time transformed runs")
    plt.ylabel("Milliseconds per puzzle")
    plt.title("Voting runtime cost")
    plt.tight_layout()
    plt.savefig(output_dir / "voting_runtime.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_accuracy_by_blanks(args.baseline, output_dir)
    save_fixed_point_failures(args.fixed_point, output_dir)
    save_voting_tradeoff(args.voting, output_dir)
    print(f"Wrote available plots to {output_dir}")


if __name__ == "__main__":
    main()

