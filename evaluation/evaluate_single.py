"""Evaluate one trained stream on its test split without ensemble fusion."""
from __future__ import annotations

import argparse

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
    get_pipeline,
    select_pipeline_input,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one trained stream on its test split."
    )
    parser.add_argument(
        "--dataset",
        choices=("xsub", "xview", "xsub120", "xset120"),
        default="xsub",
        help="Dataset/protocol to evaluate (default: xsub).",
    )
    parser.add_argument(
        "--pipeline",
        choices=("jbv", "joints", "bones", "velocity"),
        default="jbv",
        help="Single trained stream to evaluate (default: jbv).",
    )
    parser.add_argument("--run-id", default="run18_official_v2")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = get_pipeline(args.pipeline)
    num_classes = dataset_num_classes(args.dataset)
    test_dir = dataset_split_path(args.dataset, "test")
    checkpoint_dir = checkpoint_directory(args.dataset, args.run_id, pipeline)

    if not (test_dir / "binary_pt").is_dir():
        raise FileNotFoundError(f"Missing converted test split: {test_dir / 'binary_pt'}")

    test_dataset = NTUSkeletonDataset(data_folder=test_dir, max_frames=100)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=device.type == "cuda",
    )

    gcn = Spatial_GCN_Layer(in_channels=pipeline.in_channels, out_channels=128).to(device)
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
        state = torch.load(checkpoint_dir / filename, map_location=device, weights_only=True)
        module.load_state_dict(state)
        module.eval()

    correct = 0
    total = 0
    progress = tqdm(test_loader, desc=f"Testing {args.dataset} {args.pipeline}")
    with torch.no_grad():
        for data, body_mask, frame_mask, labels in progress:
            data = data.to(device)
            body_mask = body_mask.to(device)
            frame_mask = frame_mask.to(device)
            labels = labels.to(device).long().view(-1)

            batch_size, bodies, frames, joints, _ = data.shape
            stream_data = select_pipeline_input(data, pipeline)
            stream_data = stream_data.reshape(
                batch_size * bodies, frames, joints, pipeline.in_channels
            )
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
            video_representation = transformer(
                transformer_input,
                batch_size,
                bodies,
                body_mask=body_mask,
                frame_mask=frame_mask,
            )
            predictions = classifier(video_representation).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
            progress.set_postfix(accuracy=f"{100 * correct / total:.2f}%")

    if total == 0:
        raise RuntimeError(f"No test samples found in {test_dir}")
    print(f"Test accuracy ({args.dataset}, {args.pipeline}): {100 * correct / total:.2f}%")
    print(f"Correct: {correct}/{total}")


if __name__ == "__main__":
    main()
