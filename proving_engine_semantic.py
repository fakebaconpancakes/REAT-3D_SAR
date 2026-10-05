import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.dataset import NTUSkeletonDataset

# ==========================================
# 0. CONFIGURATION
# ==========================================
CURRENT_DATASET = 'X-VIEW'    
CURRENT_ENSEMBLE = '2-stream' # Change this to test different pipelines!

DATA_PREFIX = CURRENT_DATASET.lower().replace('-', '')
NPY_DIR = f"results/{DATA_PREFIX}-run17/{CURRENT_ENSEMBLE}/xai_npy"

# Kinect V2 Joint Mapping (0-indexed)
# 0-SpineBase, 1-SpineMid, 2-Neck, 3-Head, 
# 4-ShoulderL, 5-ElbowL, 6-WristL, 7-HandL, 
# 8-ShoulderR, 9-ElbowR, 10-WristR, 11-HandR, 
# 12-HipL, 13-KneeL, 14-AnkleL, 15-FootL, 
# 16-HipR, 17-KneeR, 18-AnkleR, 19-FootR, 
# 20-SpineShoulder, 21-HandTipL, 22-ThumbL, 23-HandTipR, 24-ThumbR

# Semantic Joint Groups
HEAD_GROUP = [2, 3]
ARM_L_GROUP = [4, 5, 6, 7, 21, 22]
ARM_R_GROUP = [8, 9, 10, 11, 23, 24]
ARMS_GROUP = ARM_L_GROUP + ARM_R_GROUP
LEG_L_GROUP = [12, 13, 14, 15]
LEG_R_GROUP = [16, 17, 18, 19]
LEGS_GROUP = LEG_L_GROUP + LEG_R_GROUP
CORE_GROUP = [0, 1, 20]
FULL_BODY = list(range(25))

# ==========================================
# 1. SEMANTIC BOUNDING BOXES (Action Mappings)
# ==========================================
# Map each NTU Action Index (0-59) to its logical target joint group.
# For interactions (like hugging), the whole body might be involved.
ACTION_TARGETS = {
    # Hand/Head Actions (Drinking, Brushing teeth, Phone call, etc.)
    0: ARMS_GROUP + HEAD_GROUP, 1: ARMS_GROUP + HEAD_GROUP, 2: ARMS_GROUP + HEAD_GROUP, 
    3: ARMS_GROUP + HEAD_GROUP, 17: ARMS_GROUP + HEAD_GROUP, 18: ARMS_GROUP + HEAD_GROUP,
    19: ARMS_GROUP + HEAD_GROUP, 20: ARMS_GROUP + HEAD_GROUP, 27: ARMS_GROUP + HEAD_GROUP,
    31: ARMS_GROUP + HEAD_GROUP, 35: HEAD_GROUP, 36: ARMS_GROUP + HEAD_GROUP, 
    40: HEAD_GROUP + ARMS_GROUP, 43: ARMS_GROUP + HEAD_GROUP, 47: HEAD_GROUP + CORE_GROUP,
    
    # Hand/Arm Actions (Typing, Writing, Dropping, Throwing, Clapping, etc.)
    4: ARMS_GROUP, 5: ARMS_GROUP, 6: ARMS_GROUP, 9: ARMS_GROUP, 10: ARMS_GROUP, 
    11: ARMS_GROUP, 12: ARMS_GROUP, 13: ARMS_GROUP, 14: ARMS_GROUP, 21: ARMS_GROUP, 
    22: ARMS_GROUP, 24: ARMS_GROUP, 28: ARMS_GROUP, 29: ARMS_GROUP, 30: ARMS_GROUP, 
    32: ARMS_GROUP, 33: ARMS_GROUP, 34: HEAD_GROUP + ARMS_GROUP, 37: ARMS_GROUP, 
    38: ARMS_GROUP, 39: ARMS_GROUP, 48: ARMS_GROUP, 55: ARMS_GROUP,
    
    # Leg/Foot Actions (Kicking, Hopping, Jumping, Shoe wearing, etc.)
    15: ARMS_GROUP + LEGS_GROUP, 16: ARMS_GROUP + LEGS_GROUP, 23: LEGS_GROUP, 
    25: LEGS_GROUP + CORE_GROUP, 26: LEGS_GROUP + CORE_GROUP, 41: LEGS_GROUP + CORE_GROUP, 
    42: FULL_BODY, 
    
    # Torso/Core Actions (Sitting, Standing, Backache, etc.)
    7: CORE_GROUP + LEGS_GROUP, 8: CORE_GROUP + LEGS_GROUP, 44: ARMS_GROUP + CORE_GROUP, 
    45: ARMS_GROUP + CORE_GROUP, 46: ARMS_GROUP + CORE_GROUP,
    
    # Multi-Person Interactions (Punching, Pushing, Hugging, Shaking hands)
    # For A050-A060 (indices 49-59), the interaction usually involves arms or full body dynamics.
    49: ARMS_GROUP, 50: LEGS_GROUP, 51: ARMS_GROUP, 52: ARMS_GROUP, 53: ARMS_GROUP, 
    54: ARMS_GROUP + CORE_GROUP, 55: ARMS_GROUP, 56: ARMS_GROUP, 57: ARMS_GROUP, 
    58: FULL_BODY, 59: FULL_BODY
}

# ==========================================
# 2. PROVING ENGINE LOOP
# ==========================================
dataset = NTUSkeletonDataset(data_folder=f'data/{DATA_PREFIX}/test_skeletons', max_frames=100)

hits = 0
total_evaluated = 0
results = []

print(f"Running Pointing Game for {CURRENT_ENSEMBLE} on {CURRENT_DATASET}...")

for file_idx in tqdm(range(len(dataset))):
    target_base = dataset.file_list[file_idx].replace('.pt', '')
    npy_path = os.path.join(NPY_DIR, f"{target_base}_fused.npy")
    
    if not os.path.exists(npy_path):
        continue 
        
    _, _, true_label = dataset[file_idx]
    label_idx = true_label.item()
    
    # Load the heatmap (Shape: 2 Bodies, 100 Frames, 25 Joints)
    heatmaps = np.load(npy_path) 
    
    # Find the single joint with the highest global intensity
    # Sum across frames first, then find the absolute max joint
    joint_importance = np.sum(heatmaps, axis=1) # (2, 25)
    
    # If it's a single-person action, only body 0 matters. 
    # For multi-person (A050+), the max joint across both bodies wins.
    flat_idx = np.argmax(joint_importance)
    peak_body, peak_joint = np.unravel_index(flat_idx, joint_importance.shape)
    
    # Check if it hit the semantic target
    valid_targets = ACTION_TARGETS.get(label_idx, FULL_BODY)
    
    is_hit = peak_joint in valid_targets
    
    if is_hit:
        hits += 1
    total_evaluated += 1
    
    results.append({
        'file': target_base,
        'action_idx': label_idx,
        'peak_joint': peak_joint,
        'is_hit': is_hit
    })

# ==========================================
# 3. OUTPUT RESULTS
# ==========================================
if total_evaluated > 0:
    accuracy = (hits / total_evaluated) * 100
    print("\n=== SEMANTIC FAITHFULNESS RESULTS (POINTING GAME) ===")
    print(f"Total Videos Evaluated: {total_evaluated}")
    print(f"Semantic Hits: {hits}")
    print(f"Pointing Game Accuracy: {accuracy:.2f}%")
    
    # Save the detailed breakdown
    df = pd.DataFrame(results)
    csv_path = f'results/{DATA_PREFIX}-run17/{CURRENT_ENSEMBLE}_pointing_game.csv'
    df.to_csv(csv_path, index=False)
    print(f"Detailed logs saved to {csv_path}")
else:
    print("No heatmaps found! Check your configuration.")