from dataclasses import dataclass
import hashlib
import json
import platform
from pathlib import Path
from typing import Mapping

import torch


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    channel_start: int
    channel_end: int
    checkpoint_name: str
    display_name: str

    @property
    def in_channels(self) -> int:
        return self.channel_end - self.channel_start


PIPELINES: Mapping[str, PipelineConfig] = {
    "jbv": PipelineConfig(
        name="jbv",
        channel_start=0,
        channel_end=9,
        checkpoint_name="jbv",
        display_name="Joints + Bones + Velocity",
    ),
    "joints": PipelineConfig(
        name="joints",
        channel_start=0,
        channel_end=3,
        checkpoint_name="pure_joints",
        display_name="Joints",
    ),
    "bones": PipelineConfig(
        name="bones",
        channel_start=3,
        channel_end=6,
        checkpoint_name="bones",
        display_name="Bones",
    ),
    "velocity": PipelineConfig(
        name="velocity",
        channel_start=6,
        channel_end=9,
        checkpoint_name="pure_velocity",
        display_name="Velocity",
    ),
}


DATASETS = {
    "xview": {
        "data_root": Path("data/xview_official_v2"),
        "num_classes": 60,
    },
    "xsub": {
        "data_root": Path("data/xsub_official_v2"),
        "num_classes": 60,
    },
    "xsub120": {
        "data_root": Path("data/xsub120_official_v2"),
        "num_classes": 120,
    },
    "xset120": {
        "data_root": Path("data/xset120_official_v2"),
        "num_classes": 120,
    },
}


def get_pipeline(name: str) -> PipelineConfig:
    try:
        return PIPELINES[name.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(PIPELINES))
        raise ValueError(
            f"Unknown pipeline {name!r}. Choose one of: {supported}."
        ) from error


def select_pipeline_input(
    engineered_data: torch.Tensor,
    pipeline: str | PipelineConfig,
) -> torch.Tensor:
    config = get_pipeline(pipeline) if isinstance(pipeline, str) else pipeline
    if engineered_data.shape[-1] < config.channel_end:
        raise ValueError(
            f"Pipeline {config.name!r} requires channels "
            f"{config.channel_start}:{config.channel_end}, but the input has "
            f"{engineered_data.shape[-1]} channels."
        )

    selected_data = engineered_data[..., config.channel_start:config.channel_end]
    if selected_data.shape[-1] != config.in_channels:
        raise RuntimeError(
            f"Pipeline {config.name!r} selected {selected_data.shape[-1]} "
            f"channels; expected {config.in_channels}."
        )
    return selected_data


def dataset_split_path(dataset_name: str, split: str) -> Path:
    try:
        data_root = DATASETS[dataset_name]["data_root"]
    except KeyError as error:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Choose one of: {supported}."
        ) from error

    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"Unknown split {split!r}. Choose one of: train, val, test."
        )
    return data_root / f"{split}_skeletons"


def checkpoint_directory(
    dataset_name: str,
    run_id: str,
    pipeline: str | PipelineConfig,
) -> Path:
    config = get_pipeline(pipeline) if isinstance(pipeline, str) else pipeline
    if dataset_name not in DATASETS:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Choose one of: {supported}."
        )
    if not run_id.strip():
        raise ValueError("run_id must not be empty.")
    return Path("saved_weights") / dataset_name / run_id / config.checkpoint_name


def dataset_num_classes(dataset_name: str) -> int:
    try:
        return int(DATASETS[dataset_name]["num_classes"])
    except KeyError as error:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Choose one of: {supported}."
        ) from error


def split_manifest_identifier(dataset_name: str) -> str | None:
    manifest_path = DATASETS[dataset_name]["data_root"] / "split_manifest.csv"
    if not manifest_path.is_file():
        return None
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def software_versions() -> dict[str, str | None]:
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "numpy": __import__("numpy").__version__,
        "cuda": torch.version.cuda,
    }


def ensemble_config_path(dataset_name: str, run_id: str) -> Path:
    if dataset_name not in DATASETS or not run_id.strip():
        raise ValueError("A supported dataset and non-empty run_id are required.")
    return Path("results") / dataset_name / run_id / "ensemble_config.json"


def save_ensemble_config(dataset_name: str, run_id: str, composition: list[str],
                         weights: Mapping[str, float], validation_accuracy: float) -> Path:
    if not composition or set(composition) != set(weights):
        raise ValueError("Ensemble composition must match its weight keys.")
    if any(weight < 0 for weight in weights.values()) or abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("Ensemble weights must be non-negative and sum to 1.")
    weights = {name: float(weight) for name, weight in weights.items() if weight > 0}
    composition = [name for name in composition if name in weights]
    config_path = ensemble_config_path(dataset_name, run_id)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset_name,
        "run_id": run_id,
        "composition": composition,
        "weights": dict(weights),
        "selection_split": "val",
        "validation_accuracy": float(validation_accuracy),
        "split_manifest_sha256": split_manifest_identifier(dataset_name),
    }
    temporary_path = config_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(config_path)
    return config_path


def load_ensemble_config(dataset_name: str, run_id: str) -> dict:
    config_path = ensemble_config_path(dataset_name, run_id)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Validation-selected ensemble config is missing: {config_path}. "
            "Run validation ensemble tuning first."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("dataset") != dataset_name or config.get("run_id") != run_id:
        raise ValueError(f"Ensemble config does not match {dataset_name}/{run_id}")
    if config.get("selection_split") != "val":
        raise ValueError("Ensemble configuration must be selected on validation data")
    if config.get("split_manifest_sha256") != split_manifest_identifier(dataset_name):
        raise ValueError("Ensemble config was selected against a different split manifest")
    weights = config.get("weights", {})
    if not weights or set(config.get("composition", [])) != set(weights):
        raise ValueError("Ensemble configuration has inconsistent composition and weights")
    if not set(weights).issubset({"JBV", "B", "J", "V"}):
        raise ValueError("Ensemble configuration contains an unknown stream")
    if any(float(weight) < 0 for weight in weights.values()) or abs(sum(map(float, weights.values())) - 1.0) > 1e-6:
        raise ValueError("Ensemble configuration weights must be non-negative and sum to 1")
    return config


def result_directory(dataset_name: str, run_id: str, ensemble: str) -> Path:
    if dataset_name not in DATASETS:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Choose one of: {supported}."
        )
    if not run_id.strip():
        raise ValueError("run_id must not be empty.")
    if not ensemble.strip():
        raise ValueError("ensemble must not be empty.")
    return Path("results") / dataset_name / run_id / ensemble


def write_checkpoint_metadata(
    checkpoint_dir: Path,
    dataset_name: str,
    run_id: str,
    pipeline: str | PipelineConfig,
    max_frames: int,
    num_classes: int,
    *,
    seed: int | None = None,
    deterministic: bool = False,
    best_epoch: int | None = None,
    best_val_accuracy: float | None = None,
    best_val_loss: float | None = None,
) -> None:
    config = get_pipeline(pipeline) if isinstance(pipeline, str) else pipeline
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": dataset_name,
        "run_id": run_id,
        "pipeline": config.name,
        "display_name": config.display_name,
        "in_channels": config.in_channels,
        "max_frames": max_frames,
        "num_classes": num_classes,
        "dataset_protocol": dataset_name,
        "seed": seed,
        "deterministic_algorithms": deterministic,
        "software_versions": software_versions(),
        "split_manifest_sha256": split_manifest_identifier(dataset_name),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_val_accuracy,
        "best_validation_loss": best_val_loss,
    }
    with (checkpoint_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, indent=2, sort_keys=True, fp=metadata_file)
