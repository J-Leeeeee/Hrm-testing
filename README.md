# Fixed-Point Failure Modes and Test-Time Voting in HRM Sudoku

This research fork evaluates the released 27M-parameter Hierarchical Reasoning
Model (HRM) on controlled Sudoku perturbations. It tests two questions: whether
HRM's solutions remain stable when fed back into the model, and whether
Sudoku-preserving test-time transformations can improve exact-match accuracy.

## Headline Results

The main evaluation uses the official Sudoku-Extreme checkpoint and 6,000
controlled puzzles: 1,000 uniquely solvable boards at each of six missing-cell
counts. These are controlled distribution-shift tests, not conventional Sudoku
difficulty categories.

| Missing cells | 1 | 2 | 5 | 10 | 20 | 40 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact-match accuracy | 9.0% | 16.6% | 56.7% | 92.5% | 96.9% | 100.0% |
| Invalid-output rate | 91.0% | 83.4% | 43.3% | 7.5% | 3.1% | 0.0% |

- A second inference pass regressed 3,486 of 3,717 initially correct boards
  (**93.8%**) while correcting only 14 of 2,283 initial errors (**0.6%**).
- Of the initial errors, 168 (**7.4%**) were unchanged invalid fixed points
  under the second pass.
- In a preliminary 96-puzzle test, three-way transformation voting increased
  exact-match accuracy from **62.5%** for a single run to **70.8%** with
  majority voting and **76.0%** with validity-aware reranking. This pilot result
  is reported separately from the 6,000-puzzle baseline and refeed studies.

![Accuracy and invalid-output rate by missing-cell count](figures/accuracy_by_blanks.png)

![Outcomes after feeding predictions back into HRM](figures/fixed_point_failures.png)

The reversal across missing-cell counts suggests strong sensitivity to the
checkpoint's input distribution. It should not be interpreted as evidence that
Sudoku intrinsically becomes easier when more cells are removed. See
[report.md](report.md) for methods, definitions, limitations, and complete
results.

## Reproducing the Study

Run inference on a Linux CUDA machine or CUDA-enabled WSL. CPU-only environments
are sufficient for the unit tests but not checkpoint inference.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip packaging ninja wheel setuptools setuptools-scm
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
pip install -r requirements-study.txt

huggingface-cli download sapientinc/HRM-checkpoint-sudoku-extreme --local-dir checkpoints/sudoku_extreme_hf
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000 --subsample-size 1000 --num-aug 1000
```

Smoke-test the official checkpoint:

```bash
DISABLE_COMPILE=1 OMP_NUM_THREADS=8 python evaluate.py checkpoint=checkpoints/sudoku_extreme_hf/checkpoint
```

Build the controlled datasets and reproduce the reported runs:

```bash
python -m pytest

python experiments/make_controlled_sudoku.py --limit 1000

python experiments/baseline_eval.py \
  --checkpoint checkpoints/sudoku_extreme_hf/checkpoint \
  --controlled-root data/controlled_sudoku \
  --batch-size 32 \
  --output results/controlled_baseline.csv

python experiments/fixed_point_diagnostics.py \
  --checkpoint checkpoints/sudoku_extreme_hf/checkpoint \
  --controlled-root data/controlled_sudoku \
  --batch-size 32 \
  --output results/fixed_point.csv

python experiments/test_time_voting.py \
  --checkpoint checkpoints/sudoku_extreme_hf/checkpoint \
  --controlled-root data/controlled_sudoku \
  --votes 3 \
  --limit 16 \
  --output results/voting_n16_v3.csv

python experiments/plot_results.py \
  --baseline results/controlled_baseline.csv \
  --fixed-point results/fixed_point.csv \
  --voting results/voting_n16_v3.csv
```

The inference scripts are deterministic for a fixed checkpoint, dataset, and
seed. Core artifacts are:

- `results/baseline.csv` or `results/controlled_baseline.csv`: exact accuracy, cell accuracy, validity, clue violations, runtime.
- `results/fixed_point.csv`: first prediction, refeed outcome, stable wrong and invalid fixed-point rates.
- `results/voting_n16_v3.csv`: preliminary single-run, majority-vote, and
  validity-aware reranking results.
- `figures/`: publication-ready accuracy, fixed-point, and voting-pilot plots.

Token convention used throughout the study: `PAD=0`, blank Sudoku cell
`0 -> token 1`, and digit `d -> token d+1`.

## Scope and Attribution

The HRM architecture, training pipeline, original datasets, and released
checkpoint come from [Sapient Intelligence's upstream HRM
repository](https://github.com/sapientinc/HRM). This fork adds the controlled
Sudoku generators, evaluation and transformation-voting experiments, metrics,
tests, result tables, plots, and study report. The upstream Apache 2.0 license
is retained.

---

## Upstream Project Documentation

The original HRM project README is preserved below for architecture, training,
and dataset documentation.

# Hierarchical Reasoning Model

![](./assets/hrm.png)

Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI.
Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency.
HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes.
Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities.
These results underscore HRM’s potential as a transformative advancement toward universal computation and general-purpose reasoning systems.

Read Our Paper: [https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)

**Join Our Discord Community: [https://discord.gg/sapient](https://discord.gg/sapient)**


## Quick Start Guide 🚀

### Prerequisites ⚙️

Ensure PyTorch and CUDA are installed. The repo needs CUDA extensions to be built. If not present, run the following commands:

```bash
# Install CUDA 12.6
CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda_12.6.3_560.35.05_linux.run

wget -q --show-progress --progress=bar:force:noscroll -O cuda_installer.run $CUDA_URL
sudo sh cuda_installer.run --silent --toolkit --override

export CUDA_HOME=/usr/local/cuda-12.6

# Install PyTorch with CUDA 12.6
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

pip3 install torch torchvision torchaudio --index-url $PYTORCH_INDEX_URL

# Additional packages for building extensions
pip3 install packaging ninja wheel setuptools setuptools-scm
```

Then install FlashAttention. For Hopper GPUs, install FlashAttention 3

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

For Ampere or earlier GPUs, install FlashAttention 2

```bash
pip3 install flash-attn
```

## Install Python Dependencies 🐍

```bash
pip install -r requirements.txt
```

## W&B Integration 📈

This project uses [Weights & Biases](https://wandb.ai/) for experiment tracking and metric visualization. Ensure you're logged in:

```bash
wandb login
```

## Run Experiments

### Quick Demo: Sudoku Solver 💻🗲

Train a master-level Sudoku AI capable of solving extremely difficult puzzles on a modern laptop GPU. 🧩

```bash
# Download and build Sudoku dataset
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000

# Start training (single GPU, smaller batch size)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

Runtime: ~10 hours on a RTX 4070 laptop GPU

## Trained Checkpoints 🚧

 - [ARC-AGI-2](https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2)
 - [Sudoku 9x9 Extreme (1000 examples)](https://huggingface.co/sapientinc/HRM-checkpoint-sudoku-extreme)
 - [Maze 30x30 Hard (1000 examples)](https://huggingface.co/sapientinc/HRM-checkpoint-maze-30x30-hard)

To use the checkpoints, see Evaluation section below.

## Full-scale Experiments 🔵

Experiments below assume an 8-GPU setup.

### Dataset Preparation

```bash
# Initialize submodules
git submodule update --init --recursive

# ARC-1
python dataset/build_arc_dataset.py  # ARC offical + ConceptARC, 960 examples
# ARC-2
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000  # ARC-2 official, 1120 examples

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py  # Full version
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples

# Maze
python dataset/build_maze_dataset.py  # 1000 examples
```

### Dataset Visualization

Explore the puzzles visually:

* Open `puzzle_visualizer.html` in your browser.
* Upload the generated dataset folder located in `data/...`.

## Launch experiments

### Small-sample (1K)

ARC-1:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py 
```

*Runtime:* ~24 hours

ARC-2:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/arc-2-aug-1000
```

*Runtime:* ~24 hours (checkpoint after 8 hours is often sufficient)

Sudoku Extreme (1k):

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-extreme-1k-aug-1000 epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

*Runtime:* ~10 minutes

Maze 30x30 Hard (1k):

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/maze-30x30-hard-1k epochs=20000 eval_interval=2000 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0
```

*Runtime:* ~1 hour

### Full Sudoku-Hard

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 pretrain.py data_path=data/sudoku-hard-full epochs=100 eval_interval=10 lr_min_ratio=0.1 global_batch_size=2304 lr=3e-4 puzzle_emb_lr=3e-4 weight_decay=0.1 puzzle_emb_weight_decay=0.1 arch.loss.loss_type=softmax_cross_entropy arch.L_cycles=8 arch.halt_max_steps=8 arch.pos_encodings=learned
```

*Runtime:* ~2 hours

## Evaluation

Evaluate your trained models:

* Check `eval/exact_accuracy` in W&B.
* For ARC-AGI, follow these additional steps:

```bash
OMP_NUM_THREADS=8 torchrun --nproc-per-node 8 evaluate.py checkpoint=<CHECKPOINT_PATH>
```

* Then use the provided `arc_eval.ipynb` notebook to finalize and inspect your results.

## Notes

 - Small-sample learning typically exhibits accuracy variance of around ±2 points.
 - For Sudoku-Extreme (1,000-example dataset), late-stage overfitting may cause numerical instability during training and Q-learning. It is advisable to use early stopping once the training accuracy approaches 100%.

## Citation 📜

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```
