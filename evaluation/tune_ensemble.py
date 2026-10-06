import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset
from utils.pipeline_config import (
    checkpoint_directory,
    dataset_num_classes,
    dataset_split_path,
    select_pipeline_input,
)

# 1. HARDWARE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device in-use: {device.type.upper()}")

# 2. INITIALIZATION
# IMPORTANT: Use the validation set to tune beta, NOT the test set!
# You don't want to leak test data into your hyperparameter choices.
DATASET_NAME = 'xview'
RUN_ID = 'run17'
VAL_DIR = dataset_split_path(DATASET_NAME, 'val')
BATCH_SIZE = 16
NUM_CLASSES = dataset_num_classes(DATASET_NAME)

print("Loading Data..")
val_dataset = NTUSkeletonDataset(data_folder=VAL_DIR, max_frames=100)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

print("Loading Dual-Stream Architecture...")
# ==========================================
# STREAM A: THE KINEMATIC EXPERT (9 Channels)
# ==========================================
gcn_k = Spatial_GCN_Layer(in_channels=9, out_channels=128).to(device)
transformer_k = Temporal_Brain_Layer(embed_dim=128, num_heads=4, max_frames=100, max_bodies=2).to(device)
classifier_k = nn.Linear(128, NUM_CLASSES).to(device)

jbv_dir = checkpoint_directory(DATASET_NAME, RUN_ID, 'jbv')
gcn_k.load_state_dict(torch.load(jbv_dir / 'best_gcn.pth', map_location=device, weights_only=True))
transformer_k.load_state_dict(torch.load(jbv_dir / 'best_transformer.pth', map_location=device, weights_only=True))
classifier_k.load_state_dict(torch.load(jbv_dir / 'best_classifier.pth', map_location=device, weights_only=True))
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

joints_dir = checkpoint_directory(DATASET_NAME, RUN_ID, 'joints')
gcn_b.load_state_dict(torch.load(joints_dir / 'best_gcn.pth', map_location=device, weights_only=True))
transformer_b.load_state_dict(torch.load(joints_dir / 'best_transformer.pth', map_location=device, weights_only=True))
classifier_b.load_state_dict(torch.load(joints_dir / 'best_classifier.pth', map_location=device, weights_only=True))
global_node_b = transformer_b.global_node 

gcn_b.eval()
transformer_b.eval()
classifier_b.eval()

print("Extracting Predictions...")
all_labels = []
all_probs_k = []
all_probs_b = []

loop = tqdm(val_dataloader, total=len(val_dataloader), leave=True, desc="Inferencing")
with torch.no_grad():
    for batch_idx, (batched_data, body_mask, labels) in enumerate(loop):
        batched_data = batched_data.to(device)
        body_mask = body_mask.to(device)
        labels = labels.to(device)
        all_labels.extend(labels.cpu().numpy())

        B, M, T, V, C = batched_data.shape
        
        # --- A. KINEMATIC STREAM ---
        input_k = batched_data.reshape(B * M, T, V, C)
        feat_k = gcn_k(input_k)
        frames = feat_k.shape[1]
        t_input_k = torch.cat([feat_k, global_node_k.expand(B*M, frames, 1, 128)], dim=2)
        vid_rep_k = transformer_k(t_input_k, B, M, body_mask=body_mask)
        logits_k = classifier_k(vid_rep_k)
        probs_k = torch.softmax(logits_k, dim=1) 
        all_probs_k.append(probs_k.cpu().numpy())
        
        # --- B. STRUCTURAL STREAM ---
        bone_data = select_pipeline_input(batched_data, 'bones')
        input_b = bone_data.reshape(B * M, T, V, 3)
        feat_b = gcn_b(input_b)
        t_input_b = torch.cat([feat_b, global_node_b.expand(B*M, frames, 1, 128)], dim=2)
        vid_rep_b = transformer_b(t_input_b, B, M, body_mask=body_mask)
        logits_b = classifier_b(vid_rep_b)
        probs_b = torch.softmax(logits_b, dim=1)
        all_probs_b.append(probs_b.cpu().numpy())

all_probs_k = np.concatenate(all_probs_k, axis=0)
all_probs_b = np.concatenate(all_probs_b, axis=0)
all_labels = np.array(all_labels)

print("\n" + "=" * 50)
print("🔍 GRID SEARCH: FINDING OPTIMAL ENSEMBLE WEIGHTS 🔍")
print("=" * 50)

best_acc = 0.0
best_beta = 0.0

# Test beta values from 0.0 to 1.0 in increments of 0.05
betas = np.arange(0.0, 1.05, 0.05)

for beta in betas:
    # Fused Probability = beta * Kinematic + (1 - beta) * Bone
    fused_probs = (beta * all_probs_k) + ((1.0 - beta) * all_probs_b)
    predicted_classes = np.argmax(fused_probs, axis=1)
    
    correct = np.sum(predicted_classes == all_labels)
    acc = (correct / len(all_labels)) * 100
    
    print(f"Beta: {beta:.2f} (Kinematic: {beta*100:.0f}%, Bone: {(1-beta)*100:.0f}%) -> Acc: {acc:.2f}%")
    
    if acc > best_acc:
        best_acc = acc
        best_beta = beta

print("=" * 50)
print(f"⭐ OPTIMAL ENSEMBLE RATIO FOUND! ⭐")
print(f"Best Beta: {best_beta:.2f}")
print(f"Kinematic Stream Weight: {best_beta*100:.0f}%")
print(f"Bone Stream Weight:      {(1-best_beta)*100:.0f}%")
print(f"Projected Validation Accuracy: {best_acc:.2f}%")
print("=" * 50)