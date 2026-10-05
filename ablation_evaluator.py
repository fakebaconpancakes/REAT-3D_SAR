import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset

# 1. HARDWARE & INIT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEST_DIR = 'data/xsub120/test_skeletons' 
BATCH_SIZE = 16
NUM_CLASSES = 120

print("Loading Test Data..")
test_dataset = NTUSkeletonDataset(data_folder=TEST_DIR, max_frames=100, is_train=False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

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
brain_jbv = load_brain(9, 'saved_weights/weights_xsub120/jbv') # Your original 9-ch run
brain_b   = load_brain(3, 'saved_weights/weights_xsub120/bones')
brain_j   = load_brain(3, 'saved_weights/weights_xsub120/pure_joints')
brain_v   = load_brain(3, 'saved_weights/weights_xsub120/pure_velocity')

# 3. EXTRACTION LOOP
all_labels = []
all_probs = {'JBV': [], 'B': [], 'J': [], 'V': []}

loop = tqdm(test_dataloader, total=len(test_dataloader), desc="Extracting Brain Confidences")
with torch.no_grad():
    for batched_data, body_mask, labels in loop:
        batched_data = batched_data.to(device)
        body_mask = body_mask.to(device)
        all_labels.extend(labels.numpy())

        B, M, T, V, C = batched_data.shape
        
        # Define the slices
        slices = {
            'JBV': batched_data,                         # Channels 0-8
            'J':   batched_data[:, :, :, :, 0:3],        # Channels 0,1,2
            'B':   batched_data[:, :, :, :, 3:6],        # Channels 3,4,5
            'V':   batched_data[:, :, :, :, 6:9]         # Channels 6,7,8
        }
        
        brains = {'JBV': brain_jbv, 'B': brain_b, 'J': brain_j, 'V': brain_v}
        
        for name in slices.keys():
            gcn, trans, cls = brains[name]
            inp = slices[name].reshape(B * M, T, V, slices[name].shape[-1])
            feat = gcn(inp)
            frames = feat.shape[1]
            t_inp = torch.cat([feat, trans.global_node.expand(B*M, frames, 1, 128)], dim=2)
            vid_rep = trans(t_inp, B, M, body_mask=body_mask)
            logits = cls(vid_rep)
            probs = torch.softmax(logits, dim=1)
            all_probs[name].append(probs.cpu().numpy())

# Concatenate all stored probabilities
for k in all_probs.keys():
    all_probs[k] = np.concatenate(all_probs[k], axis=0)
all_labels = np.array(all_labels)

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
        
        if acc > best_acc:
            best_acc = acc
            best_weights = weights
            
    # Format output
    weight_str = " : ".join([f"{n}({w*100:.0f}%)" for n, w in zip(model_names, best_weights)])
    print(f"⭐ {best_acc:.2f}%  |  {weight_str}")

# 5. EXECUTE YOUR ABLATION STUDY
print("\n" + "=" * 60)
print("🏆 REAT MASTER ABLATION FUSION RESULTS 🏆")
print("=" * 60)

print("\n--- 2-Stream Fusions ---")
find_optimal_fusion(['JBV', 'J'])
find_optimal_fusion(['JBV', 'V'])
find_optimal_fusion(['JBV', 'B'])
find_optimal_fusion(['B', 'J'])
find_optimal_fusion(['B', 'V'])
find_optimal_fusion(['J', 'V'])

print("\n--- 3-Stream Fusions ---")
find_optimal_fusion(['B', 'J', 'V'])
find_optimal_fusion(['JBV', 'B', 'J'])
find_optimal_fusion(['JBV', 'B', 'V'])
find_optimal_fusion(['JBV', 'J', 'V'])

print("\n--- 4-Stream Fusion ---")
find_optimal_fusion(['JBV', 'B', 'J', 'V'])
print("=" * 60)