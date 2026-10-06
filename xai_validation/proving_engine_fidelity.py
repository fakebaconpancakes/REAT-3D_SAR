import os
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset
from utils.pipeline_config import (
    checkpoint_directory,
    dataset_num_classes,
    dataset_split_path,
    result_directory,
    select_pipeline_input,
)

# ==========================================
# 0. CONFIGURATION
# ==========================================
CURRENT_DATASET = 'XSUB120'
CURRENT_ENSEMBLE = '2-stream' 
DATASET_NAME = 'xsub120'
RUN_ID = 'run17'

THESIS_CONFIGS = {
    'X-VIEW': {
        'pure':     {'JBV': 1.0,  'B': 0.0,  'J': 0.0,  'V': 0.0},
        '2-stream': {'JBV': 0.50, 'B': 0.0,  'J': 0.0,  'V': 0.50},
        '3-stream': {'JBV': 0.35, 'B': 0.0,  'J': 0.30, 'V': 0.35},
        '4-stream': {'JBV': 0.25, 'B': 0.20, 'J': 0.20, 'V': 0.35}
    },
    'X-SUB': {
        'pure':     {'JBV': 1.0,  'B': 0.0,  'J': 0.0,  'V': 0.0},
        '2-stream': {'JBV': 0.50, 'B': 0.0,  'J': 0.0,  'V': 0.50},
        '3-stream': {'JBV': 0.35, 'B': 0.35, 'J': 0.0,  'V': 0.30},
        '4-stream': {'JBV': 0.30, 'B': 0.20, 'J': 0.20, 'V': 0.30}
    },
    'XSET120': {
        'pure':     {'JBV': 1.0,  'B': 0.0,  'J': 0.0,  'V': 0.0},
        '2-stream': {'JBV': 0.55, 'B': 0.0,  'J': 0.0,  'V': 0.45},
        '3-stream': {'JBV': 0.40, 'B': 0.0,  'J': 0.25, 'V': 0.35},
        '4-stream': {'JBV': 0.30, 'B': 0.20, 'J': 0.20, 'V': 0.30}
    },
    'XSUB120': {
        'pure':     {'JBV': 1.0,  'B': 0.0,  'J': 0.0,  'V': 0.0},
        '2-stream': {'JBV': 0.55, 'B': 0.0,  'J': 0.45, 'V': 0.0},
        '3-stream': {'JBV': 0.35, 'B': 0.0,  'J': 0.30, 'V': 0.35},
        '4-stream': {'JBV': 0.30, 'B': 0.20, 'J': 0.20, 'V': 0.30}
    }
}

WEIGHTS = THESIS_CONFIGS[CURRENT_DATASET][CURRENT_ENSEMBLE]
ACTIVE_STREAMS = [name for name, weight in WEIGHTS.items() if weight > 0.0]
NUM_CLASSES = dataset_num_classes(DATASET_NAME)
RESULT_DIR = result_directory(DATASET_NAME, RUN_ID, CURRENT_ENSEMBLE)
NPY_DIR = RESULT_DIR / "xai_npy"
FIDELITY_DIR = RESULT_DIR / "fidelity"
FIDELITY_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [1, 3, 5, 7, 9] # How many joints to delete/insert

# ==========================================
# 1. LOAD THE COMMITTEE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def load_expert(in_channels, path):
    gcn = Spatial_GCN_Layer(in_channels=in_channels, out_channels=128).to(device)
    trans = Temporal_Brain_Layer(embed_dim=128, num_heads=4, max_frames=100, max_bodies=2).to(device)
    cls = nn.Linear(128, NUM_CLASSES).to(device) 
    
    base_weight_path = checkpoint_directory(DATASET_NAME, RUN_ID, {
        'jbv': 'jbv',
        'bones': 'bones',
        'pure_joints': 'joints',
        'pure_velocity': 'velocity',
    }[path])
    gcn.load_state_dict(torch.load(base_weight_path / 'best_gcn.pth', map_location=device, weights_only=True))
    trans.load_state_dict(torch.load(base_weight_path / 'best_transformer.pth', map_location=device, weights_only=True))
    cls.load_state_dict(torch.load(base_weight_path / 'best_classifier.pth', map_location=device, weights_only=True))
    
    gcn.eval(); trans.eval(); cls.eval()
    return gcn, trans, cls

experts = {}
if 'JBV' in ACTIVE_STREAMS: experts['JBV'] = load_expert(9, 'jbv')
if 'B' in ACTIVE_STREAMS:   experts['B']   = load_expert(3, 'bones')
if 'J' in ACTIVE_STREAMS:   experts['J']   = load_expert(3, 'pure_joints')
if 'V' in ACTIVE_STREAMS:   experts['V']   = load_expert(3, 'pure_velocity')

# ==========================================
# 2. INFERENCE FUNCTION
# ==========================================
def get_ensemble_confidence(tensor_input, body_mask, target_label):
    """Runs the 9-channel tensor through the active ensemble and returns confidence of the TRUE label."""
    B, M, T, V, C = tensor_input.shape
    slices = {
        'JBV': select_pipeline_input(tensor_input, 'jbv'),
        'J': select_pipeline_input(tensor_input, 'joints'),
        'B': select_pipeline_input(tensor_input, 'bones'),
        'V': select_pipeline_input(tensor_input, 'velocity'),
    }
    
    fused_probs = torch.zeros(1, NUM_CLASSES).to(device)
    
    with torch.no_grad():
        for name in ACTIVE_STREAMS:
            gcn, trans, cls = experts[name]
            inp = slices[name].reshape(B * M, T, V, slices[name].shape[-1])
            feat = gcn(inp)
            frames = feat.shape[1]
            t_inp = torch.cat([feat, trans.global_node.expand(B*M, frames, 1, 128)], dim=2)
            video_rep = trans(t_inp, B, M, body_mask=body_mask, return_attention=False)
            probs = torch.softmax(cls(video_rep), dim=1)
            fused_probs += WEIGHTS[name] * probs 
            
    # Return the probability of the GROUND TRUTH label
    return fused_probs[0, target_label].item()

# ==========================================
# 3. PROVING ENGINE LOOP
# ==========================================
dataset = NTUSkeletonDataset(
    data_folder=str(dataset_split_path(DATASET_NAME, 'test')),
    max_frames=100,
)
results = []

print(f"Running Proving Engine ({CURRENT_ENSEMBLE} on {CURRENT_DATASET})...")
for file_idx in tqdm(range(len(dataset))):
    target_base = dataset.file_list[file_idx].replace('.pt', '')
    npy_path = NPY_DIR / f"{target_base}_fused.npy"
    
    if not os.path.exists(npy_path):
        continue # Only evaluate videos that have XAI heatmaps generated
        
    engineered_tensor, body_mask, true_label = dataset[file_idx]
    tensor_input = engineered_tensor.unsqueeze(0).to(device)
    body_mask = body_mask.unsqueeze(0).to(device)
    label_idx = true_label.item()
    
    # 1. Base Confidence (No Masking)
    base_conf = get_ensemble_confidence(tensor_input, body_mask, label_idx)
    
    # 2. Rank Joints by XAI Importance
    heatmaps = np.load(npy_path) # Shape: (2, 100, 25)
    joint_importance = np.sum(heatmaps, axis=1) # Sum across frames -> (2, 25)
    
    video_metrics = {'sample_id': target_base, 'base_conf': base_conf}
    
    for k in K_VALUES:
        del_tensor = tensor_input.clone()
        ins_tensor = torch.zeros_like(tensor_input)
        
        for m in range(2):
            # Get indices of the top k joints for this body
            top_k_joints = np.argsort(joint_importance[m])[-k:]
            
            for j in range(25):
                if j in top_k_joints:
                    # Deletion: Mute the top joints
                    del_tensor[0, m, :, j, :] = 0.0
                    # Insertion: Keep only the top joints
                    ins_tensor[0, m, :, j, :] = tensor_input[0, m, :, j, :]
                else:
                    # Deletion: Keep the non-important joints
                    pass 
                    # Insertion: non-important joints remain 0.0
                    
        # Calculate new confidences
        del_conf = get_ensemble_confidence(del_tensor, body_mask, label_idx)
        ins_conf = get_ensemble_confidence(ins_tensor, body_mask, label_idx)
        
        video_metrics[f'del_k{k}'] = del_conf
        video_metrics[f'ins_k{k}'] = ins_conf
        
    results.append(video_metrics)

# ==========================================
# 4. AGGREGATE AND SAVE
# ==========================================
df = pd.DataFrame(results)
details_path = FIDELITY_DIR / "fidelity_details.csv"
df.to_csv(details_path, index=False)

# Calculate Area Under Curve (AUC) for Drops
print("\n=== QUANTITATIVE FAITHFULNESS RESULTS ===")
base_confidence_mean = float(df["base_conf"].mean()) if not df.empty else 0.0
print(f"Base Average Confidence: {base_confidence_mean:.4f}")

deletion_drop_mean = {}
insertion_retention_mean = {}
for k in K_VALUES:
    del_drop = (
        base_confidence_mean - float(df[f"del_k{k}"].mean())
        if not df.empty
        else 0.0
    )
    ins_retention = (
        float(df[f"ins_k{k}"].mean()) / base_confidence_mean
        if base_confidence_mean > 0 and not df.empty
        else 0.0
    )
    deletion_drop_mean[str(k)] = del_drop
    insertion_retention_mean[str(k)] = ins_retention
    print(f"Top {k} Joints -> F_del Drop: -{del_drop:.4f} | F_ins Retention: {ins_retention*100:.1f}%")

fidelity_result = {
    "schema_version": 1,
    "metric": "deletion_insertion_fidelity",
    "dataset": DATASET_NAME,
    "run_id": RUN_ID,
    "ensemble": CURRENT_ENSEMBLE,
    "sample_count": len(results),
    "k_values": K_VALUES,
    "aggregate": {
        "base_confidence_mean": base_confidence_mean,
        "deletion_drop_mean": deletion_drop_mean,
        "insertion_retention_mean": insertion_retention_mean,
    },
    "details_file": "fidelity_details.csv",
}
with (FIDELITY_DIR / "fidelity_results.json").open("w", encoding="utf-8") as result_file:
    json.dump(fidelity_result, result_file, indent=2, sort_keys=True)

print(f"Saved fidelity details to {details_path}.")