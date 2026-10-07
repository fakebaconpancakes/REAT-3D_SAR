import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.dataset import NTUSkeletonDataset
from utils.action_labels import NTU_ACTION_LABELS
from utils.pipeline_config import dataset_split_path, result_directory

# ==========================================
# 0. CONFIGURATION
# ==========================================
CURRENT_DATASET = 'X-SUB'    
CURRENT_ENSEMBLE = '4-stream' # Change this to test different pipelines!
DATASET_NAME = 'xsub'
RUN_ID = 'weights_xsub-run17'

RESULT_DIR = result_directory(DATASET_NAME, RUN_ID, CURRENT_ENSEMBLE)
NPY_DIR = RESULT_DIR / "xai_npy"
SEMANTIC_DIR = RESULT_DIR / "semantic"
SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)

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
JOINT_NAMES = [
    "SpineBase", "SpineMid", "Neck", "Head", "ShoulderL", "ElbowL",
    "WristL", "HandL", "ShoulderR", "ElbowR", "WristR", "HandR",
    "HipL", "KneeL", "AnkleL", "FootL", "HipR", "KneeR", "AnkleR",
    "FootR", "SpineShoulder", "HandTipL", "ThumbL", "HandTipR", "ThumbR",
]

# These groups are disjoint so their attribution shares add to 1.0.
ANATOMICAL_GROUPS = {
    "head": HEAD_GROUP,
    "left_arm": ARM_L_GROUP,
    "right_arm": ARM_R_GROUP,
    "core": CORE_GROUP,
    "left_leg": LEG_L_GROUP,
    "right_leg": LEG_R_GROUP,
}

# ==========================================
# 1. SEMANTIC BOUNDING BOXES (Action Mappings)
# ==========================================
# The classifier labels are zero-based: A1 is index 0 and A120 is index 119.
# Each action has an explicit target group; no 120-class action falls through
# to FULL_BODY because it is missing from the semantic map.
ACTION_TARGETS = {
    # Daily actions A1-A40.
    0: ARMS_GROUP + HEAD_GROUP,
    1: ARMS_GROUP + HEAD_GROUP,
    2: ARMS_GROUP + HEAD_GROUP,
    3: ARMS_GROUP + HEAD_GROUP,
    4: ARMS_GROUP,
    5: ARMS_GROUP + CORE_GROUP,
    6: ARMS_GROUP,
    7: CORE_GROUP + LEGS_GROUP,
    8: CORE_GROUP + LEGS_GROUP,
    9: ARMS_GROUP,
    10: ARMS_GROUP + HEAD_GROUP,
    11: ARMS_GROUP,
    12: ARMS_GROUP,
    13: ARMS_GROUP + HEAD_GROUP,
    14: ARMS_GROUP + HEAD_GROUP,
    15: ARMS_GROUP + LEGS_GROUP,
    16: ARMS_GROUP + LEGS_GROUP,
    17: ARMS_GROUP + HEAD_GROUP,
    18: ARMS_GROUP + HEAD_GROUP,
    19: ARMS_GROUP + HEAD_GROUP,
    20: ARMS_GROUP + HEAD_GROUP,
    21: ARMS_GROUP,
    22: ARMS_GROUP,
    23: LEGS_GROUP,
    24: ARMS_GROUP + CORE_GROUP,
    25: LEGS_GROUP + CORE_GROUP,
    26: LEGS_GROUP + CORE_GROUP,
    27: ARMS_GROUP + HEAD_GROUP,
    28: ARMS_GROUP + HEAD_GROUP,
    29: ARMS_GROUP,
    30: ARMS_GROUP,
    31: ARMS_GROUP + HEAD_GROUP,
    32: ARMS_GROUP,
    33: HEAD_GROUP,
    34: HEAD_GROUP,
    35: ARMS_GROUP + HEAD_GROUP,
    36: ARMS_GROUP + HEAD_GROUP,
    37: ARMS_GROUP,
    38: ARMS_GROUP,
    39: ARMS_GROUP,
    # Medical conditions A41-A49.
    40: ARMS_GROUP + HEAD_GROUP,
    41: FULL_BODY,
    42: FULL_BODY,
    43: HEAD_GROUP,
    44: CORE_GROUP,
    45: CORE_GROUP,
    46: HEAD_GROUP + CORE_GROUP,
    47: FULL_BODY,
    48: ARMS_GROUP + HEAD_GROUP,
    # Mutual actions A50-A60.
    49: ARMS_GROUP,
    50: LEGS_GROUP,
    51: ARMS_GROUP + CORE_GROUP,
    52: ARMS_GROUP + CORE_GROUP,
    53: ARMS_GROUP,
    54: ARMS_GROUP + CORE_GROUP,
    55: ARMS_GROUP,
    56: ARMS_GROUP,
    57: FULL_BODY,
    58: FULL_BODY,
    59: FULL_BODY,
    # Daily actions A61-A102.
    60: ARMS_GROUP + HEAD_GROUP,
    61: ARMS_GROUP + HEAD_GROUP,
    62: ARMS_GROUP + LEGS_GROUP,
    63: ARMS_GROUP + LEGS_GROUP,
    64: ARMS_GROUP + LEGS_GROUP,
    65: ARMS_GROUP,
    66: ARMS_GROUP + HEAD_GROUP,
    67: ARMS_GROUP + HEAD_GROUP,
    68: ARMS_GROUP,
    69: ARMS_GROUP,
    70: ARMS_GROUP,
    71: ARMS_GROUP,
    72: ARMS_GROUP,
    73: ARMS_GROUP,
    74: ARMS_GROUP,
    75: ARMS_GROUP,
    76: ARMS_GROUP,
    77: ARMS_GROUP,
    78: HEAD_GROUP + ARMS_GROUP,
    79: LEGS_GROUP + CORE_GROUP,
    80: ARMS_GROUP,
    81: ARMS_GROUP,
    82: ARMS_GROUP,
    83: ARMS_GROUP,
    84: ARMS_GROUP + HEAD_GROUP,
    85: ARMS_GROUP,
    86: ARMS_GROUP + CORE_GROUP,
    87: ARMS_GROUP + CORE_GROUP,
    88: ARMS_GROUP,
    89: ARMS_GROUP,
    90: ARMS_GROUP,
    91: FULL_BODY,
    92: ARMS_GROUP + CORE_GROUP,
    93: ARMS_GROUP,
    94: ARMS_GROUP + CORE_GROUP,
    95: ARMS_GROUP,
    96: ARMS_GROUP,
    97: LEGS_GROUP + CORE_GROUP,
    98: LEGS_GROUP,
    99: LEGS_GROUP,
    100: LEGS_GROUP + CORE_GROUP,
    101: LEGS_GROUP,
    # Medical conditions A103-A105.
    102: HEAD_GROUP,
    103: FULL_BODY,
    104: ARMS_GROUP + HEAD_GROUP,
    # Mutual actions A106-A120.
    105: ARMS_GROUP,
    106: ARMS_GROUP,
    107: ARMS_GROUP + CORE_GROUP,
    108: ARMS_GROUP,
    109: ARMS_GROUP + CORE_GROUP,
    110: LEGS_GROUP,
    111: ARMS_GROUP,
    112: ARMS_GROUP + HEAD_GROUP,
    113: ARMS_GROUP + CORE_GROUP,
    114: ARMS_GROUP + HEAD_GROUP,
    115: FULL_BODY,
    116: HEAD_GROUP + ARMS_GROUP,
    117: ARMS_GROUP,
    118: ARMS_GROUP + CORE_GROUP,
    119: ARMS_GROUP,
}

if set(ACTION_TARGETS) != set(range(120)):
    missing = sorted(set(range(120)) - set(ACTION_TARGETS))
    extra = sorted(set(ACTION_TARGETS) - set(range(120)))
    raise RuntimeError(
        f"Semantic map must contain A1-A120 exactly; missing={missing}, extra={extra}."
    )
if sorted(joint for joints in ANATOMICAL_GROUPS.values() for joint in joints) != list(range(25)):
    raise RuntimeError("Anatomical groups must partition all 25 joints exactly once.")


def target_group_names(target_joints):
    return [
        group_name
        for group_name, joints in ANATOMICAL_GROUPS.items()
        if set(target_joints).intersection(joints)
    ]


def calculate_attribution_confidence(heatmaps, target_joints):
    """Return global group shares and within-group joint shares.

    Heatmaps contain non-negative relative attribution scores, not probabilities.
    A share is therefore calculated as attribution mass divided by total mass.
    """
    if heatmaps.ndim != 3 or heatmaps.shape[-1] != 25:
        raise ValueError(
            f"Expected heatmap shape (bodies, frames, 25), got {heatmaps.shape}."
        )

    joint_mass = np.sum(heatmaps, axis=(0, 1))
    total_mass = float(np.sum(joint_mass))
    if total_mass <= 0:
        raise ValueError("Cannot calculate attribution confidence for an empty heatmap.")

    group_confidence = {
        group_name: float(np.sum(joint_mass[joints]) / total_mass)
        for group_name, joints in ANATOMICAL_GROUPS.items()
    }
    joint_confidence = {
        joint_name: float(joint_mass[joint_idx] / total_mass)
        for joint_idx, joint_name in enumerate(JOINT_NAMES)
    }
    joint_confidence_within_group = {}
    for group_name, joints in ANATOMICAL_GROUPS.items():
        group_mass = float(np.sum(joint_mass[joints]))
        joint_confidence_within_group[group_name] = {
            JOINT_NAMES[joint_idx]: (
                float(joint_mass[joint_idx] / group_mass)
                if group_mass > 0
                else 0.0
            )
            for joint_idx in joints
        }

    return {
        "group_confidence": group_confidence,
        "joint_confidence": joint_confidence,
        "joint_confidence_within_group": joint_confidence_within_group,
        "target_group_confidence": float(
            np.sum(joint_mass[target_joints]) / total_mass
        ),
        "total_attribution_mass": total_mass,
    }

# ==========================================
# 2. PROVING ENGINE LOOP
# ==========================================
dataset = NTUSkeletonDataset(
    data_folder=str(dataset_split_path(DATASET_NAME, 'test')),
    max_frames=100,
)

hits = 0
total_evaluated = 0
results = []
group_totals = {group_name: 0.0 for group_name in ANATOMICAL_GROUPS}

print(f"Running Pointing Game for {CURRENT_ENSEMBLE} on {CURRENT_DATASET}...")

for file_idx in tqdm(range(len(dataset))):
    target_base = dataset.file_list[file_idx].replace('.pt', '')
    npy_path = NPY_DIR / f"{target_base}_fused.npy"
    
    if not os.path.exists(npy_path):
        continue 
        
    _, _, true_label = dataset[file_idx]
    label_idx = int(true_label.item())
    
    # Load the heatmap (Shape: 2 Bodies, 100 Frames, 25 Joints)
    heatmaps = np.load(npy_path) 
    
    # Find the single joint with the highest global intensity
    # Sum across frames first, then find the absolute max joint
    joint_importance = np.sum(heatmaps, axis=1) # (2, 25)
    
    # If it's a single-person action, only body 0 matters. 
    # For multi-person (A050+), the max joint across both bodies wins.
    flat_idx = np.argmax(joint_importance)
    peak_body, peak_joint = (
        int(index) for index in np.unravel_index(flat_idx, joint_importance.shape)
    )
    
    valid_targets = ACTION_TARGETS[label_idx]
    attribution = calculate_attribution_confidence(heatmaps, valid_targets)
    for group_name, confidence in attribution["group_confidence"].items():
        group_totals[group_name] += confidence
    
    is_hit = peak_joint in valid_targets
    
    if is_hit:
        hits += 1
    total_evaluated += 1
    
    results.append({
        'sample_id': target_base,
        'action_idx': label_idx,
        'action_label': NTU_ACTION_LABELS[label_idx],
        'peak_body': peak_body,
        'peak_joint': peak_joint,
        'peak_joint_name': JOINT_NAMES[peak_joint],
        'is_hit': is_hit
    })
    results[-1].update({
        "target_groups": target_group_names(valid_targets),
        "target_group_confidence": attribution["target_group_confidence"],
        "group_confidence": attribution["group_confidence"],
        "joint_confidence": attribution["joint_confidence"],
        "joint_confidence_within_group": attribution[
            "joint_confidence_within_group"
        ],
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
    csv_path = SEMANTIC_DIR / "semantic_details.csv"
    df.to_csv(csv_path, index=False)
    print(f"Detailed logs saved to {csv_path}")
else:
    accuracy = 0.0
    csv_path = SEMANTIC_DIR / "semantic_details.csv"
    pd.DataFrame(
        columns=[
            "sample_id",
            "action_idx",
            "action_label",
            "peak_body",
            "peak_joint",
            "peak_joint_name",
            "is_hit",
            "target_group_confidence",
        ]
    ).to_csv(csv_path, index=False)
    print("No heatmaps found. Check your configuration.")

semantic_result = {
    "schema_version": 2,
    "metric": "pointing_game",
    "dataset": DATASET_NAME,
    "run_id": RUN_ID,
    "ensemble": CURRENT_ENSEMBLE,
    "sample_count": total_evaluated,
    "hits": hits,
    "accuracy": accuracy / 100,
    "confidence_definition": {
        "name": "attribution_mass_share",
        "formula": (
            "confidence(group) = sum(H[b,t,j] for j in group) / "
            "sum(H[b,t,j] for all b,t,j)"
        ),
        "joint_formula": (
            "confidence(joint) = sum(H[b,t,j]) / "
            "sum(H[b,t,j] for all b,t,j)"
        ),
        "within_group_joint_formula": (
            "confidence(joint | group) = sum(H[b,t,j]) / "
            "sum(H[b,t,k] for k in group)"
        ),
        "interpretation": (
            "These are relative XAI attribution shares, not model class "
            "probabilities or causal probabilities."
        ),
    },
    "group_confidence_mean": {
        group_name: (
            group_totals[group_name] / total_evaluated
            if total_evaluated > 0
            else 0.0
        )
        for group_name in ANATOMICAL_GROUPS
    },
    "details_json_file": "semantic_details.json",
    "details_file": "semantic_details.csv",
}
with (SEMANTIC_DIR / "semantic_results.json").open("w", encoding="utf-8") as result_file:
    json.dump(semantic_result, result_file, indent=2, sort_keys=True)
with (SEMANTIC_DIR / "semantic_details.json").open("w", encoding="utf-8") as details_file:
    json.dump(
        {
            "schema_version": 2,
            "joint_names": JOINT_NAMES,
            "anatomical_groups": ANATOMICAL_GROUPS,
            "samples": results,
        },
        details_file,
        indent=2,
        sort_keys=True,
    )