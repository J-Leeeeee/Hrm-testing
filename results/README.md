# Result Artifacts

The committed CSVs contain aggregate-safe evaluation records only; checkpoints
and generated puzzle arrays are excluded from the repository.

| File | Scope | Rows |
| --- | --- | ---: |
| `controlled_baseline.csv` | One-pass evaluation, 1,000 puzzles at each of six missing-cell counts | 6,000 |
| `fixed_point.csv` | One-pass plus prediction-refeed outcomes on the same controlled sets | 6,000 |
| `voting_n16_v3.csv` | Three-run voting pilot, 96 puzzles recorded once for each of three selection methods | 288 |

The main baseline and fixed-point CSVs support the claims in the repository
README and `report.md`. The smaller voting CSV is deliberately labeled as a
pilot and should not be treated as a 6,000-puzzle result.

Large or interrupted voting sweeps use the local `voting.csv` filename, which
is ignored until a complete run has been validated and intentionally promoted
to a versioned result artifact.
