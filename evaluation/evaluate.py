"""Evaluate validation-selected 1-, 2-, 3-, and 4-stream ensembles on test."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset
from utils.pipeline_config import (
    checkpoint_directory,
    dataset_num_classes,
    dataset_split_path,
    load_ensemble_config,
    select_pipeline_input,
    split_manifest_identifier,
)


# Change these for a different trained experiment, or edit them before running.
DATASET_NAME = "xsub"
RUN_ID = "run18_official_v2"
BATCH_SIZE = 16
STREAM_PIPELINES = {"JBV": "jbv", "B": "bones", "J": "joints", "V": "velocity"}


def selected_ensembles(config: dict) -> dict[str, dict]:
    """Read new per-size configs, while accepting older single-config files."""
    if "best_by_stream_count" in config:
        return config["best_by_stream_count"]
    count = str(len(config["composition"]))
    return {
        count: {
            "composition": config["composition"],
            "weights": config["weights"],
            "validation_accuracy": config["validation_accuracy"],
        }
    }


def load_brain(stream_name: str, num_classes: int, device: torch.device):
    pipeline_name = STREAM_PIPELINES[stream_name]
    pipeline = pipeline_name
    in_channels = 9 if stream_name == "JBV" else 3
    folder_path = checkpoint_directory(DATASET_NAME, RUN_ID, pipeline)
    gcn = Spatial_GCN_Layer(in_channels=in_channels, out_channels=128).to(device)
    transformer = Temporal_Brain_Layer(
        embed_dim=128,
        num_heads=4,
        max_frames=100,
        max_bodies=2,
    ).to(device)
    classifier = nn.Linear(128, num_classes).to(device)

    for module, filename in (
        (gcn, "best_gcn.pth"),
        (transformer, "best_transformer.pth"),
        (classifier, "best_classifier.pth"),
    ):
        module.load_state_dict(
            torch.load(folder_path / filename, map_location=device, weights_only=True)
        )
        module.eval()
    return gcn, transformer, classifier


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = dataset_num_classes(DATASET_NAME)
    config = load_ensemble_config(DATASET_NAME, RUN_ID)
    ensembles = selected_ensembles(config)
    needed_streams = sorted(
        {name for result in ensembles.values() for name in result["composition"]}
    )

    test_dir = dataset_split_path(DATASET_NAME, "test")
    if not (test_dir / "binary_pt").is_dir():
        raise FileNotFoundError(f"Missing converted test split: {test_dir / 'binary_pt'}")
    dataset = NTUSkeletonDataset(data_folder=test_dir, max_frames=100)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=device.type == "cuda",
    )

    print(f"Device: {device.type.upper()}")
    print(f"Dataset/run: {DATASET_NAME}/{RUN_ID}")
    print(f"Loading streams needed for selected ensembles: {needed_streams}")
    experts = {name: load_brain(name, num_classes, device) for name in needed_streams}
    correct_by_count = {count: 0 for count in ensembles}
    total = 0

    with torch.no_grad():
        for data, body_mask, frame_mask, labels in tqdm(loader, desc="Test evaluation"):
            data = data.to(device)
            body_mask = body_mask.to(device)
            frame_mask = frame_mask.to(device)
            labels = labels.to(device).long().view(-1)
            batch_size, bodies, frames, joints, _ = data.shape
            probabilities_by_stream = {}

            for name in needed_streams:
                pipeline = STREAM_PIPELINES[name]
                stream_data = select_pipeline_input(data, pipeline)
                stream_data = stream_data.reshape(
                    batch_size * bodies, frames, joints, stream_data.shape[-1]
                )
                gcn, transformer, classifier = experts[name]
                features = gcn(stream_data)
                feature_frames = features.shape[1]
                transformer_input = torch.cat(
                    [
                        features,
                        transformer.global_node.expand(
                            batch_size * bodies, feature_frames, 1, 128
                        ),
                    ],
                    dim=2,
                )
                representation = transformer(
                    transformer_input,
                    batch_size,
                    bodies,
                    body_mask=body_mask,
                    frame_mask=frame_mask,
                )
                probabilities_by_stream[name] = torch.softmax(
                    classifier(representation), dim=1
                )

            for count, result in ensembles.items():
                fused = torch.zeros(batch_size, num_classes, device=device)
                for name in result["composition"]:
                    fused += float(result["weights"][name]) * probabilities_by_stream[name]
                correct_by_count[count] += (fused.argmax(dim=1) == labels).sum().item()
            total += labels.numel()

    if total == 0:
        raise RuntimeError(f"No test samples found in {test_dir}")

    results = {}
    for count, selected in sorted(ensembles.items(), key=lambda item: int(item[0])):
        results[count] = {
            "label": "pure single-stream" if count == "1" else f"best {count}-stream ensemble",
            "composition": selected["composition"],
            "weights": selected["weights"],
            "validation_accuracy_pct": float(selected["validation_accuracy"]),
            "test_accuracy_pct": 100.0 * correct_by_count[count] / total,
            "correct": int(correct_by_count[count]),
            "total": int(total),
        }

    # This summary is identified using validation results only; test scores are reporting metrics.
    best_validation_count = max(
        ensembles,
        key=lambda count: float(ensembles[count]["validation_accuracy"]),
    )
    report = {
        "dataset": DATASET_NAME,
        "run_id": RUN_ID,
        "selection_split": "val",
        "evaluation_split": "test",
        "split_manifest_sha256": split_manifest_identifier(DATASET_NAME),
        "test_samples": int(total),
        "best_overall_by_validation": best_validation_count,
        "test_used_for_selection": False,
        "results_by_stream_count": results,
    }
    report_path = Path("results") / DATASET_NAME / RUN_ID / "test_evaluation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = report_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_path.replace(report_path)

    print("\nTest results by ensemble size:")
    for count, result in results.items():
        print(
            f"{result['label']}: {result['test_accuracy_pct']:.2f}% | "
            f"composition={result['composition']} | weights={result['weights']}"
        )
    print(f"Saved detailed report: {report_path}")


if __name__ == "__main__":
    main()
