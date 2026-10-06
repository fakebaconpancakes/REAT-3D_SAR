from dataclasses import dataclass
import json
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
        "data_root": Path("data/xview"),
        "num_classes": 60,
    },
    "xsub": {
        "data_root": Path("data/xsub"),
        "num_classes": 60,
    },
    "xsub120": {
        "data_root": Path("data/xsub120"),
        "num_classes": 120,
    },
    "xset120": {
        "data_root": Path("data/xset120"),
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
    }
    with (checkpoint_dir / "metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, indent=2, sort_keys=True, fp=metadata_file)
