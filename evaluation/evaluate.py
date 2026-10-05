import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb # <-- Added WandB

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset
from utils.xai_extractor import extract_xai_red_dots

# 0. INITIALIZE WANDB
wandb.init(
    project="HAR-REAT",
    name="EVAL-2-STREAM-FUSION", 
    config={
        "architecture": "Late Fusion (9-Ch Kinematic + 3-Ch Structural)",
        "dataset": "NTU-RGB+D X-View (Pure Test Set)",
        "batch_size": 16
    }
)

# 1. HARDWARE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in-use: {device.type.upper()}")

# 2. INITIALIZATION
VAL_DIR = 'data/xview/test_skeletons' # <-- Updated to isolated Test Set
BATCH_SIZE = 16
NUM_CLASSES = 60

print("Loading Data..")
# The dataset returns the full 9-channel tensor
val_dataset = NTUSkeletonDataset(data_folder=VAL_DIR, max_frames=100, is_train=False)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

print("Loading Dual-Stream Architecture...")
# ==========================================
# STREAM A: THE KINEMATIC EXPERT (9 Channels)
# ==========================================
gcn_k = Spatial_GCN_Layer(in_channels=9, out_channels=128).to(device)
transformer_k = Temporal_Brain_Layer(embed_dim=128, num_heads=4, max_frames=100, max_bodies=2).to(device)
classifier_k = nn.Linear(128, NUM_CLASSES).to(device)

gcn_k.load_state_dict(torch.load('saved_weights/joints/best_gcn.pth', map_location=device, weights_only=True))
transformer_k.load_state_dict(torch.load('saved_weights/joints/best_transformer.pth', map_location=device, weights_only=True))
classifier_k.load_state_dict(torch.load('saved_weights/joints/best_classifier.pth', map_location=device, weights_only=True))
global_node_k = transformer_k.global_node 

gcn_k.eval()
transformer_k.eval()
classifier_k.eval()

# ==========================================
# STREAM B: THE STRUCTURAL EXPERT (3 Channels)
# ==========================================
gcn_b = Spatial_GCN_Layer(in_channels=3, out_channels=128).to(device)
transformer_b = Temporal_Brain_Layer(embed_dim=128, num_heads=4, max_frames=100, max_bodies=2).to(device)
classifier_b = nn.Linear(128, NUM_CLASSES).to(device)

gcn_b.load_state_dict(torch.load('saved_weights/bones/best_gcn.pth', map_location=device, weights_only=True))
transformer_b.load_state_dict(torch.load('saved_weights/bones/best_transformer.pth', map_location=device, weights_only=True))
classifier_b.load_state_dict(torch.load('saved_weights/bones/best_classifier.pth', map_location=device, weights_only=True))
global_node_b = transformer_b.global_node 

gcn_b.eval()
transformer_b.eval()
classifier_b.eval()

# 4. Evaluation Loop
print("Running 2-Stream Ensemble Evaluation..")

total_samples = 0
correct_predictions = 0

loop = tqdm(val_dataloader, total=len(val_dataloader), leave=True, desc="Evaluating")
with torch.no_grad():
    for batch_idx, (batched_data, body_mask, labels) in enumerate(loop):
        batched_data = batched_data.to(device)
        body_mask = body_mask.to(device)
        labels = labels.to(device)

        B, M, T, V, C = batched_data.shape
        
        # --- A. KINEMATIC STREAM (Uses all 9 channels) ---
        input_k = batched_data.reshape(B * M, T, V, C)
        feat_k = gcn_k(input_k)
        frames = feat_k.shape[1]
        t_input_k = torch.cat([feat_k, global_node_k.expand(B*M, frames, 1, 128)], dim=2)
        vid_rep_k = transformer_k(t_input_k, B, M, body_mask=body_mask)
        logits_k = classifier_k(vid_rep_k)
        probs_k = torch.softmax(logits_k, dim=1) 
        
        # --- B. STRUCTURAL STREAM (Slices out only channels 3, 4, and 5: The Bones!) ---
        bone_data = batched_data[:, :, :, :, 3:6]
        input_b = bone_data.reshape(B * M, T, V, 3)
        feat_b = gcn_b(input_b)
        t_input_b = torch.cat([feat_b, global_node_b.expand(B*M, frames, 1, 128)], dim=2)
        vid_rep_b = transformer_b(t_input_b, B, M, body_mask=body_mask)
        logits_b = classifier_b(vid_rep_b)
        probs_b = torch.softmax(logits_b, dim=1) 
        
        # --- C. HYBRID LATE FUSION ---
        fused_probs = (0.5 * probs_k) + (0.5 * probs_b)
        
        _, predicted_classes = torch.max(fused_probs, 1)
        predicted_classes = predicted_classes.view(-1)
        labels = labels.long().view(-1) 
        
        correct_predictions += torch.eq(predicted_classes, labels).sum().item()
        total_samples += labels.size(0)

        current_acc = (correct_predictions / total_samples) * 100
        loop.set_postfix(acc=f"{current_acc:.2f}%")
        
        # Log the running accuracy to WandB
        wandb.log({"Running Accuracy (%)": current_acc})

# 5. Final Report
accuracy = (correct_predictions / total_samples) * 100

# Log final metrics to WandB
wandb.log({"Final Ensemble Accuracy (%)": accuracy, "Total Unseen Videos": total_samples})
wandb.finish()

print("\n" + "=" * 50)
print("🏆 2-STREAM ENSEMBLE EVALUATION COMPLETE 🏆")
print(f"Total Unseen Videos Processed: {total_samples}")
print(f"Final Top-1 Accuracy:          {accuracy:.2f}%")
print("=" * 50)