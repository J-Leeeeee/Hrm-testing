from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: Iterable[dict[str, object]], *, group_key: str | None = None) -> None:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(group_key, "all")) if group_key is not None else "all"
        groups[key].append(row)

    for key, group in sorted(groups.items()):
        n = len(group)
        if n == 0:
            continue
        exact = sum(float(row.get("exact", 0)) for row in group) / n
        cell = sum(float(row.get("cell_accuracy", 0.0)) for row in group) / n
        invalid = sum(float(row.get("invalid", 0)) for row in group) / n
        clue = sum(float(row.get("clue_violations", 0)) for row in group) / n
        print(
            f"{key}: n={n} exact={exact:.4f} cell={cell:.4f} "
            f"invalid={invalid:.4f} clue_violations={clue:.3f}"
        )

