import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset
from train60 import NUM_CLASSES

# ==========================================
# 0. CONFIGURATION
# ==========================================
CURRENT_DATASET = 'X-VIEW'    
CURRENT_ENSEMBLE = '2-stream' 

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
    }
}

WEIGHTS = THESIS_CONFIGS[CURRENT_DATASET][CURRENT_ENSEMBLE]
ACTIVE_STREAMS = [name for name, weight in WEIGHTS.items() if weight > 0.0]
DATA_PREFIX = CURRENT_DATASET.lower().replace('-', '')
NPY_DIR = f"results/{DATA_PREFIX}-run17/{CURRENT_ENSEMBLE}/xai_npy"

K_VALUES = [1, 3, 5, 7, 9] # How many joints to delete/insert

# ==========================================
# 1. LOAD THE COMMITTEE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def load_expert(in_channels, path):
    gcn = Spatial_GCN_Layer(in_channels=in_channels, out_channels=128).to(device)
    trans = Temporal_Brain_Layer(embed_dim=128, num_heads=4, max_frames=100, max_bodies=2).to(device)
    cls = nn.Linear(128, NUM_CLASSES).to(device) 
    
    base_weight_path = f'saved_weights/weights_{DATA_PREFIX}-run17/{path}'
    gcn.load_state_dict(torch.load(f'{base_weight_path}/best_gcn.pth', map_location=device, weights_only=True))
    trans.load_state_dict(torch.load(f'{base_weight_path}/best_transformer.pth', map_location=device, weights_only=True))
    cls.load_state_dict(torch.load(f'{base_weight_path}/best_classifier.pth', map_location=device, weights_only=True))
    
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
        'JBV': tensor_input,
        'J':   tensor_input[:, :, :, :, 0:3],
        'B':   tensor_input[:, :, :, :, 3:6],
        'V':   tensor_input[:, :, :, :, 6:9]
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
dataset = NTUSkeletonDataset(data_folder=f'data/{DATA_PREFIX}/test_skeletons', max_frames=100)
results = []

print(f"Running Proving Engine ({CURRENT_ENSEMBLE} on {CURRENT_DATASET})...")
for file_idx in tqdm(range(len(dataset))):
    target_base = dataset.file_list[file_idx].replace('.pt', '')
    npy_path = os.path.join(NPY_DIR, f"{target_base}_fused.npy")
    
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
    
    video_metrics = {'file': target_base, 'base_conf': base_conf}
    
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

# Calculate Area Under Curve (AUC) for Drops
print("\n=== QUANTITATIVE FAITHFULNESS RESULTS ===")
print(f"Base Average Confidence: {df['base_conf'].mean():.4f}")

for k in K_VALUES:
    del_drop = df['base_conf'].mean() - df[f'del_k{k}'].mean()
    ins_retention = df[f'ins_k{k}'].mean() / df['base_conf'].mean()
    print(f"Top {k} Joints -> F_del Drop: -{del_drop:.4f} | F_ins Retention: {ins_retention*100:.1f}%")

df.to_csv(f'results/{DATA_PREFIX}-run17/{CURRENT_ENSEMBLE}_fidelity_metrics.csv', index=False)
print("Saved raw metrics to CSV.")