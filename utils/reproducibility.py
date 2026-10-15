"""Seed control and resumable training state helpers."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    if deterministic:
        # CUBLAS reads this on its first CUDA context/matmul initialization.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)


def seed_data_loader_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def capture_rng_state(generator: torch.Generator) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_loader": generator.get_state(),
    }


def restore_rng_state(state: dict, generator: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["cuda"]])
    generator.set_state(state["data_loader"].cpu())


def save_training_state(
    path: Path,
    epoch: int,
    models: dict[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    best_epoch: int,
    best_val_accuracy: float,
    best_val_loss: float,
    generator: torch.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "models": {name: model.state_dict() for name, model in models.items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_epoch": best_epoch,
            "best_val_accuracy": best_val_accuracy,
            "best_val_loss": best_val_loss,
            "rng_state": capture_rng_state(generator),
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_training_state(path: Path, device: torch.device, models: dict[str, torch.nn.Module],
                        optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler,
                        generator: torch.Generator) -> dict:
    state = torch.load(path, map_location=device, weights_only=False)
    for name, model in models.items():
        model.load_state_dict(state["models"][name])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng_state"], generator)
    return state
