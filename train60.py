import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import os
import argparse
import json

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset
from utils.pipeline_config import (
    checkpoint_directory,
    dataset_num_classes,
    dataset_split_path,
    get_pipeline,
    select_pipeline_input,
    write_checkpoint_metadata,
    software_versions,
    split_manifest_identifier,
)
from utils.reproducibility import (
    load_training_state,
    save_training_state,
    seed_data_loader_worker,
    seed_everything,
)

# =====================
# 1. HYPERPARAMETERS
# =====================
DATASET_NAME = "xsub"
RUN_ID = "run18_official_v2"
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_CLASSES = dataset_num_classes(DATASET_NAME)

def parse_args():
    parser = argparse.ArgumentParser(description="Train an NTU-RGB+D 60 pipeline.")
    parser.add_argument(
        "--pipeline",
        choices=("jbv", "joints", "bones", "velocity"),
        default="jbv",
        help="Input representation to train.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model and data-loader initialization.")
    parser.add_argument("--deterministic", action="store_true", help="Request deterministic PyTorch algorithms.")
    parser.add_argument("--resume", action="store_true", help="Resume from this run's last_training_state.pth.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    pipeline = get_pipeline(args.pipeline)
    data_dir = dataset_split_path(DATASET_NAME, "train")
    val_dir = dataset_split_path(DATASET_NAME, "val")
    checkpoint_dir = checkpoint_directory(DATASET_NAME, RUN_ID, pipeline)
    for split_path in (data_dir, val_dir):
        if not (split_path / "binary_pt").is_dir():
            raise FileNotFoundError(
                f"Missing binary dataset split: {split_path / 'binary_pt'}. "
                "Training requires both train and validation data."
            )
    manifest_id = split_manifest_identifier(DATASET_NAME)
    if manifest_id is None:
        raise FileNotFoundError(f"Missing split_manifest.csv for {DATASET_NAME}; rebuild the corrected split first.")

    wandb.init(
        project="HAR-REAT",
        name=f"{RUN_ID}-{DATASET_NAME}-{pipeline.name}",
        config={
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "architecture": "Spatial GCN + Transformer",
            "dataset": "NTU-RGB+D",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "scheduler": "10-Epoch Linear Warmup + Cosine Decay",
            "pipeline": pipeline.name,
            "pipeline_display_name": pipeline.display_name,
            "pipeline_channels": pipeline.in_channels,
            "checkpoint_dir": str(checkpoint_dir),
            "seed": args.seed,
            "deterministic_algorithms": args.deterministic,
            "software_versions": software_versions(),
            "split_manifest_sha256": manifest_id,
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f'Device Type: {device.type.upper()}')
    print(f"Pipeline: {pipeline.display_name} ({pipeline.in_channels} channels)")
    print(f"Training data: {data_dir}")
    print(f"Validation data: {val_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")

    # =====================
    # 2. INTIALIZATION
    # =====================
    print("Loading Dataset..")
    dataset = NTUSkeletonDataset(data_folder=str(data_dir), max_frames=100)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True,
        persistent_workers=True, worker_init_fn=seed_data_loader_worker, generator=loader_generator,
    )

    print("Loading Validation Dataset..")
    val_dataset = NTUSkeletonDataset(data_folder=str(val_dir), max_frames=100)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # The Neural Networks
    gcn = Spatial_GCN_Layer(in_channels=pipeline.in_channels, out_channels=128).to(device)
    transformer = Temporal_Brain_Layer(embed_dim=128, num_heads=4, max_frames=100, max_bodies=2).to(device)
    global_node = transformer.global_node

    # The Classifier
    classifier = nn.Linear(128, NUM_CLASSES).to(device)

    wandb.watch(gcn, log="all", log_freq=10)
    wandb.watch(transformer, log="all", log_freq=10)

    # Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        list(gcn.parameters()) + list(transformer.parameters()) + list(classifier.parameters()),
        lr = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY
    )

    # --- UPGRADE: Linear Warmup + Cosine Annealing ---
    warmup_epochs = 10
    cosine_epochs = EPOCHS - warmup_epochs

    # 1. Warmup: Start at 1% of the LR, ramp up to 100% over 10 epochs
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)

    # 2. Cosine Decay: Take over at Epoch 11, curve down to near-zero by Epoch 100
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-6)

    # 3. Stitch them together
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # =====================
    # 3. TRAINING LOOP
    # =====================
    print("Start Training..")

    gcn.train()
    transformer.train()
    classifier.train()
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    start_epoch = 0
    models = {"gcn": gcn, "transformer": transformer, "classifier": classifier}
    if args.resume:
        resume_path = checkpoint_dir / "last_training_state.pth"
        if not resume_path.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {resume_path}")
        metadata_path = checkpoint_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Cannot verify resume metadata: {metadata_path}")
        run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_resume = {
            "dataset_protocol": DATASET_NAME,
            "run_id": RUN_ID,
            "seed": args.seed,
            "deterministic_algorithms": args.deterministic,
            "split_manifest_sha256": manifest_id,
        }
        for key, value in expected_resume.items():
            if run_metadata.get(key) != value:
                raise ValueError(f"Resume metadata mismatch for {key}: {run_metadata.get(key)!r} != {value!r}")
        resume_state = load_training_state(resume_path, device, models, optimizer, scheduler, loader_generator)
        start_epoch = int(resume_state["epoch"])
        best_val_acc = float(resume_state["best_val_accuracy"])
        best_val_loss = float(resume_state["best_val_loss"])
        best_epoch = int(resume_state["best_epoch"])
        print(f"Resuming at epoch {start_epoch + 1}; best validation accuracy is {best_val_acc:.2f}%")

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        loop = tqdm(dataloader, total=len(dataloader), leave=True, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for batch_idx, (batched_data, body_mask, frame_mask, labels) in enumerate(loop):
            batched_data = batched_data.to(device)
            body_mask = body_mask.to(device)
            frame_mask = frame_mask.to(device)
            labels = labels.to(device)

            B, M, T, V, C = batched_data.shape
            selected_batch = select_pipeline_input(batched_data, pipeline)
            gcn_input = selected_batch.reshape(B*M, T, V, pipeline.in_channels)

            # 1. Forward Pass (GCN)
            gcn_features = gcn(gcn_input) # Output: (Batch*2, Time, Joints, 64)

            # 2. Attach Global Node
            frames = gcn_features.shape[1]
            global_node_expanded = global_node.expand(B*M, frames, 1, 128)
            transformer_input = torch.cat([gcn_features, global_node_expanded], dim=2)

            # 3. Forward Pass (Transformer)
            video_representation = transformer(transformer_input, B, M, body_mask=body_mask, frame_mask=frame_mask)

            # 5. Predictions
            predictions = classifier(video_representation)

            # 6. Loss
            loss = criterion(predictions, labels)

            # 7. Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted_classes = torch.max(predictions, 1)
            correct_predictions += (predicted_classes == labels).sum().item()
            total_samples += labels.size(0)

            loop.set_postfix(loss=loss.item())

        # =====================
        # 4. VALIDATION LOOP
        # =====================
        gcn.eval()
        transformer.eval()
        classifier.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        
        with torch.no_grad(): # Turns off gradient tracking to save memory/speed
            for val_batch, val_mask, val_frame_mask, val_labels in val_dataloader:
                val_batch = val_batch.to(device)
                val_mask = val_mask.to(device)
                val_frame_mask = val_frame_mask.to(device)
                val_labels = val_labels.to(device)

                
                v_B, v_M, v_T, v_V, v_C = val_batch.shape

                # Validation Folding Trick
                selected_val_batch = select_pipeline_input(val_batch, pipeline)
                val_gcn_input = selected_val_batch.reshape(v_B*v_M, v_T, v_V, pipeline.in_channels)
                v_gcn_feat = gcn(val_gcn_input)
                
                v_frames = v_gcn_feat.shape[1]
                v_global_node = global_node.expand(v_B*v_M, v_frames, 1, 128)
                v_transformer_input = torch.cat([v_gcn_feat, v_global_node], dim=2)

                # Validation Intercation Call
                v_vid_rep = transformer(v_transformer_input, v_B, v_M, body_mask=val_mask, frame_mask=val_frame_mask)
                
                v_preds = classifier(v_vid_rep)
                v_loss = criterion(v_preds, val_labels)
                
                val_loss += v_loss.item()
                _, v_pred_classes = torch.max(v_preds, 1)
                val_correct += (v_pred_classes == val_labels).sum().item()
                val_samples += val_labels.size(0)

        epoch_val_loss = val_loss / len(val_dataloader)
        epoch_val_acc = (val_correct / val_samples) * 100

        if best_epoch == 0 or epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            write_checkpoint_metadata(
                checkpoint_dir,
                DATASET_NAME,
                RUN_ID,
                pipeline,
                max_frames=100,
                num_classes=NUM_CLASSES,
                seed=args.seed,
                deterministic=args.deterministic,
                best_epoch=best_epoch,
                best_val_accuracy=best_val_acc,
                best_val_loss=best_val_loss,
            )
            torch.save(gcn.state_dict(), checkpoint_dir / 'best_gcn.pth')
            torch.save(transformer.state_dict(), checkpoint_dir / 'best_transformer.pth')
            torch.save(classifier.state_dict(), checkpoint_dir / 'best_classifier.pth')
            print(f"🌟 New Best Model! Saved with Val Acc: {best_val_acc:.2f}%")

        # Put models back into training mode for the next epoch!
        gcn.train()
        transformer.train()
        classifier.train()

        # Step the Scheduler at the end of the Epoch
        scheduler.step()
        
        # Extract current Learning Rate to track it
        current_lr = optimizer.param_groups[0]['lr']

        # Print Epoch Report
        epoch_accuracy = (correct_predictions / total_samples) * 100
        epoch_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | LR: {current_lr:.6f} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_accuracy:.2f}% | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")
        
        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": epoch_loss,
            "Train Accuracy": epoch_accuracy,
            "Validation Loss": epoch_val_loss,
            "Validation Accuracy": epoch_val_acc,
            "Learning Rate": current_lr
        })

        # Save weights every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(gcn.state_dict(), checkpoint_dir / f'gcn_epoch_{epoch+1}.pth')
            torch.save(transformer.state_dict(), checkpoint_dir / f'transformer_epoch_{epoch+1}.pth')
            torch.save(classifier.state_dict(), checkpoint_dir / f'classifier_epoch_{epoch+1}.pth')
            print(f"-> Checkpoint saved for Epoch {epoch+1}")

        save_training_state(
            checkpoint_dir / "last_training_state.pth",
            epoch + 1,
            models,
            optimizer,
            scheduler,
            best_epoch,
            best_val_acc,
            best_val_loss,
            loader_generator,
        )

    print("Training Complete!")
