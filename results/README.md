# Result Artifacts

The committed CSVs contain aggregate-safe evaluation records only; checkpoints
and generated puzzle arrays are excluded from the repository.

| File | Scope | Rows |
| --- | --- | ---: |
| `controlled_baseline.csv` | One-pass evaluation, 1,000 puzzles at each of six missing-cell counts | 6,000 |
| `fixed_point.csv` | One-pass plus prediction-refeed outcomes on the same controlled sets | 6,000 |
| `voting.csv` | Test-time voting on the same 6,000 boards at 1, 3, 5, and 10 votes, recorded for `single`, `majority`, and `rerank` | 72,000 |
| `voting_n16_v3.csv` | Superseded 96-puzzle, 3-vote pilot. Do not cite as the main voting result. | 288 |

`controlled_baseline.csv`, `fixed_point.csv`, and `voting.csv` are the artifacts
behind the claims in the repository README and `report.md`. Pilot CSVs named
`*_n64.csv` stay local and gitignored.
