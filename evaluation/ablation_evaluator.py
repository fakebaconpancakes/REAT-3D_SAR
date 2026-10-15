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
    save_ensemble_config,
)

# 1. HARDWARE & INIT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_NAME = 'xsub120'
RUN_ID = 'run18_official_v2'
VAL_DIR = dataset_split_path(DATASET_NAME, 'val')
BATCH_SIZE = 16
NUM_CLASSES = dataset_num_classes(DATASET_NAME)

print("Loading Validation Data..")
val_dataset = NTUSkeletonDataset(data_folder=VAL_DIR, max_frames=100)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

# 2. HELPER TO LOAD A BRAIN
def load_brain(in_channels, folder_path):
    gcn = Spatial_GCN_Layer(in_channels=in_channels, out_channels=128).to(device)
    transformer = Temporal_Brain_Layer(embed_dim=128, num_heads=4, max_frames=100, max_bodies=2).to(device)
    classifier = nn.Linear(128, NUM_CLASSES).to(device)
    
    gcn.load_state_dict(torch.load(f'{folder_path}/best_gcn.pth', map_location=device, weights_only=True))
    transformer.load_state_dict(torch.load(f'{folder_path}/best_transformer.pth', map_location=device, weights_only=True))
    classifier.load_state_dict(torch.load(f'{folder_path}/best_classifier.pth', map_location=device, weights_only=True))
    
    gcn.eval(); transformer.eval(); classifier.eval()
    return gcn, transformer, classifier

print("Loading All 4 Brains...")
brain_jbv = load_brain(
    9,
    checkpoint_directory(DATASET_NAME, RUN_ID, 'jbv'),
)
brain_b = load_brain(
    3,
    checkpoint_directory(DATASET_NAME, RUN_ID, 'bones'),
)
brain_j = load_brain(
    3,
    checkpoint_directory(DATASET_NAME, RUN_ID, 'joints'),
)
brain_v = load_brain(
    3,
    checkpoint_directory(DATASET_NAME, RUN_ID, 'velocity'),
)

# 3. EXTRACTION LOOP
all_labels = []
all_probs = {'JBV': [], 'B': [], 'J': [], 'V': []}

loop = tqdm(val_dataloader, total=len(val_dataloader), desc="Extracting Validation Brain Confidences")
with torch.no_grad():
    for batched_data, body_mask, frame_mask, labels in loop:
        batched_data = batched_data.to(device)
        body_mask = body_mask.to(device)
        frame_mask = frame_mask.to(device)
        all_labels.extend(labels.numpy())

        # Each stream is inferred from this shared batch, keeping predictions aligned by sample.

        B, M, T, V, C = batched_data.shape
        
        # Define the slices
        slices = {
            'JBV': select_pipeline_input(batched_data, 'jbv'),
            'J': select_pipeline_input(batched_data, 'joints'),
            'B': select_pipeline_input(batched_data, 'bones'),
            'V': select_pipeline_input(batched_data, 'velocity'),
        }
        
        brains = {'JBV': brain_jbv, 'B': brain_b, 'J': brain_j, 'V': brain_v}
        
        for name in slices.keys():
            gcn, trans, cls = brains[name]
            inp = slices[name].reshape(B * M, T, V, slices[name].shape[-1])
            feat = gcn(inp)
            frames = feat.shape[1]
            t_inp = torch.cat([feat, trans.global_node.expand(B*M, frames, 1, 128)], dim=2)
            vid_rep = trans(t_inp, B, M, body_mask=body_mask, frame_mask=frame_mask)
            logits = cls(vid_rep)
            probs = torch.softmax(logits, dim=1)
            all_probs[name].append(probs.cpu().numpy())

# Concatenate all stored probabilities
for k in all_probs.keys():
    all_probs[k] = np.concatenate(all_probs[k], axis=0)
all_labels = np.array(all_labels)
if len(all_labels) != len(val_dataset) or any(len(probabilities) != len(all_labels) for probabilities in all_probs.values()):
    raise RuntimeError("Validation predictions are not aligned sample-for-sample")

# 4. GRID SEARCH ENGINE
def find_optimal_fusion(model_names, step=0.05):
    best_acc = 0.0
    best_weights = None
    num_models = len(model_names)
    
    # Generate weight combinations that sum to 1.0
    valid_weights = []
    if num_models == 2:
        for w1 in np.arange(0, 1.01, step):
            valid_weights.append([w1, 1.0 - w1])
    elif num_models == 3:
        for w1 in np.arange(0, 1.01, step):
            for w2 in np.arange(0, 1.01 - w1, step):
                valid_weights.append([w1, w2, 1.0 - w1 - w2])
    elif num_models == 4:
        for w1 in np.arange(0, 1.01, step):
            for w2 in np.arange(0, 1.01 - w1, step):
                for w3 in np.arange(0, 1.01 - w1 - w2, step):
                    valid_weights.append([w1, w2, w3, 1.0 - w1 - w2 - w3])

    for weights in valid_weights:
        # Multiply each brain's probability by its weight and sum them
        fused = np.zeros_like(all_probs[model_names[0]])
        for idx, name in enumerate(model_names):
            fused += weights[idx] * all_probs[name]
            
        preds = np.argmax(fused, axis=1)
        acc = np.mean(preds == all_labels) * 100
        
        if best_weights is None or acc > best_acc:
            best_acc = acc
            best_weights = weights
            
    # Format output
    weight_str = " : ".join([f"{n}({w*100:.0f}%)" for n, w in zip(model_names, best_weights)])
    print(f"⭐ {best_acc:.2f}%  |  {weight_str}")

    return best_acc, dict(zip(model_names, map(float, best_weights)))


# 5. EXECUTE YOUR ABLATION STUDY
print("\n" + "=" * 60)
print("🏆 REAT MASTER ABLATION FUSION RESULTS 🏆")
print("=" * 60)

best_overall = None


def report_candidate(model_names):
    global best_overall
    accuracy, weights = find_optimal_fusion(model_names)
    if best_overall is None or accuracy > best_overall["accuracy"]:
        best_overall = {"accuracy": accuracy, "composition": model_names, "weights": weights}


print("\n--- 2-Stream Fusions ---")
report_candidate(['JBV', 'J'])
report_candidate(['JBV', 'V'])
report_candidate(['JBV', 'B'])
report_candidate(['B', 'J'])
report_candidate(['B', 'V'])
report_candidate(['J', 'V'])

print("\n--- 3-Stream Fusions ---")
report_candidate(['B', 'J', 'V'])
report_candidate(['JBV', 'B', 'J'])
report_candidate(['JBV', 'B', 'V'])
report_candidate(['JBV', 'J', 'V'])

print("\n--- 4-Stream Fusion ---")
report_candidate(['JBV', 'B', 'J', 'V'])
if best_overall is None:
    raise RuntimeError("No validation ensemble candidates were evaluated")
config_path = save_ensemble_config(
    DATASET_NAME,
    RUN_ID,
    best_overall["composition"],
    best_overall["weights"],
    best_overall["accuracy"],
)
print(f"Best validation composition: {best_overall['composition']}")
print(f"Saved validation-selected ensemble config: {config_path}")
print("=" * 60)
