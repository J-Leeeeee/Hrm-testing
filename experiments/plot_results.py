from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


OUTCOME_LABELS = {
    "became_correct": "Became correct",
    "became_wrong": "Became wrong",
    "changed_wrong": "Changed, still wrong",
    "invalid_fixed_point": "Invalid fixed point",
    "stable_correct": "Stable correct",
    "stable_wrong": "Stable wrong",
}

METHOD_LABELS = {
    "single": "Single",
    "majority": "Majority vote",
    "rerank": "Validity-aware rerank",
}


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

    plt.figure(figsize=(8, 4.8))
    sns.lineplot(data=summary, x="blank_count", y="exact_accuracy", marker="o", label="Exact")
    sns.lineplot(data=summary, x="blank_count", y="cell_accuracy", marker="o", label="Cell")
    sns.lineplot(data=summary, x="blank_count", y="invalid_rate", marker="o", label="Invalid")
    plt.xlabel("Missing cells")
    plt.ylabel("Rate")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.ylim(-0.02, 1.02)
    plt.title("Accuracy rises as more cells are missing")
    plt.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_by_blanks.png", dpi=200, bbox_inches="tight")
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
    summary["Outcome"] = summary["outcome"].map(OUTCOME_LABELS).fillna(summary["outcome"])

    hue_order = [
        OUTCOME_LABELS[outcome]
        for outcome in ("became_wrong", "changed_wrong", "invalid_fixed_point", "stable_correct", "became_correct")
        if outcome in set(summary["outcome"])
    ]
    plt.figure(figsize=(9, 5))
    sns.barplot(data=summary, x="blank_count", y="rate", hue="Outcome", hue_order=hue_order)
    plt.xlabel("Missing cells")
    plt.ylabel("Outcome rate")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.ylim(0, 1)
    plt.title("Second-pass outcomes by missing-cell count")
    plt.legend(title=None, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_dir / "fixed_point_failures.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_voting_tradeoff(voting_path: str | Path, output_dir: Path) -> None:
    if not has_data(voting_path):
        return
    df = pd.read_csv(voting_path)
    if df.empty:
        return
    key_columns = [column for column in ("dataset", "example_id") if column in df.columns]
    if len(key_columns) == 2 and df["votes"].nunique() > 1:
        key_sets = []
        for vote_count in sorted(df["votes"].unique()):
            vote_rows = df[(df["votes"] == vote_count) & (df["method"] == "single")]
            key_sets.append(set(map(tuple, vote_rows[key_columns].itertuples(index=False, name=None))))
        common_keys = set.intersection(*key_sets)
        row_keys = pd.MultiIndex.from_frame(df[key_columns])
        common_index = pd.MultiIndex.from_tuples(sorted(common_keys), names=key_columns)
        df = df[row_keys.isin(common_index)]

    summary = df.groupby(["votes", "method"], as_index=False).agg(
        exact_accuracy=("exact", "mean"),
        invalid_rate=("invalid", "mean"),
        elapsed_ms=("elapsed_ms", "mean"),
    )
    summary["Method"] = summary["method"].map(METHOD_LABELS).fillna(summary["method"])
    puzzle_count = df[df["method"] == "single"][key_columns].drop_duplicates().shape[0] if len(key_columns) == 2 else 0

    plt.figure(figsize=(8, 4.8))
    if summary["votes"].nunique() == 1:
        method_order = [method for method in ("single", "majority", "rerank") if method in set(summary["method"])]
        ax = sns.barplot(data=summary, x="method", y="exact_accuracy", hue="method", order=method_order, hue_order=method_order, legend=False)
        for container in ax.containers:
            ax.bar_label(container, labels=[f"{bar.get_height():.1%}" for bar in container], padding=3)
        ax.set_xticks(range(len(method_order)), ["Single", "Majority\nvote", "Validity-aware\nrerank"][: len(method_order)])
        vote_count = int(summary["votes"].iloc[0])
        plt.xlabel("")
        plt.title(f"{vote_count}-run voting pilot (n={puzzle_count} puzzles)")
    else:
        sns.lineplot(data=summary, x="votes", y="exact_accuracy", hue="Method", marker="o")
        plt.xlabel("Test-time transformed runs")
        plt.title(f"Voting accuracy on {puzzle_count} common puzzles")
        plt.legend(title=None, frameon=False)
    plt.ylabel("Exact-match accuracy")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.ylim(0, 1.02)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_dir / "voting_vs_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close()

    if summary["votes"].nunique() > 1:
        runtime = summary.groupby("votes", as_index=False)["elapsed_ms"].mean()
        plt.figure(figsize=(7.5, 4.5))
        sns.lineplot(data=runtime, x="votes", y="elapsed_ms", marker="o")
        plt.xlabel("Test-time transformed runs")
        plt.ylabel("Milliseconds per puzzle")
        plt.title(f"Observed voting latency on {puzzle_count} common puzzles")
        sns.despine()
        plt.tight_layout()
        plt.savefig(output_dir / "voting_runtime.png", dpi=200, bbox_inches="tight")
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    save_accuracy_by_blanks(args.baseline, output_dir)
    save_fixed_point_failures(args.fixed_point, output_dir)
    save_voting_tradeoff(args.voting, output_dir)
    print(f"Wrote available plots to {output_dir}")


if __name__ == "__main__":
    main()

