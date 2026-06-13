from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from .constants import IGNORE_LABEL_ID, PAD_TOKEN, SEQ_LEN


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.common import PuzzleDatasetMetadata  # noqa: E402
from pretrain import PretrainConfig, init_train_state  # noqa: E402


def resolve_checkpoint_file(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    if path.is_file():
        return path
    if path.is_dir():
        direct = path / "checkpoint"
        if direct.exists():
            return direct
        step_files = sorted(path.glob("step_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if step_files:
            return step_files[0]
    raise FileNotFoundError(f"Could not find checkpoint file from {checkpoint!s}.")


def load_train_metadata(data_path: str | Path) -> PuzzleDatasetMetadata:
    metadata_path = Path(data_path) / "train" / "dataset.json"
    if not metadata_path.exists():
        metadata_path = Path(data_path) / "test" / "dataset.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return PuzzleDatasetMetadata(**yaml.safe_load(f))


@dataclass
class HRMRunner:
    model: torch.nn.Module
    batch_size: int
    device: torch.device
    checkpoint_file: Path

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def predict(self, inputs: np.ndarray, labels: np.ndarray | None = None, puzzle_identifiers: np.ndarray | None = None) -> np.ndarray:
        inputs = np.asarray(inputs, dtype=np.int64)
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        if inputs.ndim != 2 or inputs.shape[1] != SEQ_LEN:
            raise ValueError(f"Expected inputs shaped [N, {SEQ_LEN}], got {inputs.shape}.")
        if len(inputs) > self.batch_size:
            raise ValueError(f"Runner batch_size={self.batch_size} cannot handle batch of {len(inputs)}.")

        real_size = len(inputs)
        pad_size = self.batch_size - real_size

        if labels is None:
            labels = np.full_like(inputs, IGNORE_LABEL_ID)
        else:
            labels = np.asarray(labels, dtype=np.int64)
            if labels.ndim == 1:
                labels = labels.reshape(1, -1)

        if puzzle_identifiers is None:
            puzzle_identifiers = np.zeros((real_size,), dtype=np.int64)
        else:
            puzzle_identifiers = np.asarray(puzzle_identifiers, dtype=np.int64).reshape(-1)

        if pad_size:
            inputs = np.pad(inputs, ((0, pad_size), (0, 0)), constant_values=PAD_TOKEN)
            labels = np.pad(labels, ((0, pad_size), (0, 0)), constant_values=IGNORE_LABEL_ID)
            puzzle_identifiers = np.pad(puzzle_identifiers, (0, pad_size), constant_values=0)

        batch = {
            "inputs": torch.as_tensor(inputs, dtype=torch.int32, device=self.device),
            "labels": torch.as_tensor(labels, dtype=torch.int32, device=self.device),
            "puzzle_identifiers": torch.as_tensor(puzzle_identifiers, dtype=torch.int32, device=self.device),
        }

        with torch.inference_mode():
            with torch.device(self.device):
                carry = self.model.initial_carry(batch)  # type: ignore[attr-defined]

            while True:
                carry, _, _, preds, all_finish = self.model(
                    carry=carry,
                    batch=batch,
                    return_keys=["logits"],
                )
                if all_finish:
                    break

            logits = preds["logits"]
            pred_tokens = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int16)

        return pred_tokens[:real_size]


def load_runner(
    *,
    checkpoint: str | Path,
    data_path: str | Path,
    batch_size: int,
    disable_compile: bool = True,
) -> HRMRunner:
    if not torch.cuda.is_available():
        raise RuntimeError("HRM inference requires CUDA. Run these scripts on a Linux cloud GPU or CUDA-enabled WSL.")

    if disable_compile:
        os.environ.setdefault("DISABLE_COMPILE", "1")

    checkpoint_file = resolve_checkpoint_file(checkpoint)
    config_path = checkpoint_file.parent / "all_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected checkpoint config at {config_path}.")

    with config_path.open("r", encoding="utf-8") as f:
        config = PretrainConfig(**yaml.safe_load(f))

    config.data_path = str(data_path)
    config.global_batch_size = batch_size
    config.checkpoint_path = str(checkpoint_file.parent)
    config.eval_save_outputs = ["logits"]

    train_metadata = load_train_metadata(data_path)
    train_state = init_train_state(config, train_metadata, world_size=1)

    state_dict = torch.load(checkpoint_file, map_location="cuda")
    try:
        train_state.model.load_state_dict(state_dict, assign=True)
    except Exception:
        stripped = {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}
        train_state.model.load_state_dict(stripped, assign=True)

    train_state.model.eval()
    return HRMRunner(
        model=train_state.model,
        batch_size=batch_size,
        device=torch.device("cuda"),
        checkpoint_file=checkpoint_file,
    )

