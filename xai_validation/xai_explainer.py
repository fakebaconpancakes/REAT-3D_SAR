import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import argparse
import torch.nn as nn

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer
from utils.dataset import NTUSkeletonDataset
from utils.xai_extractor import apply_differential_xai, normalize_global_heatmap
from utils.action_labels import NTU_ACTION_LABELS
from utils.pipeline_config import (
    checkpoint_directory,
    dataset_num_classes,
    dataset_split_path,
    result_directory,
    select_pipeline_input,
)

NTU_CLASSES = NTU_ACTION_LABELS


def action_name(label_index):
    if 0 <= label_index < len(NTU_CLASSES):
        return NTU_CLASSES[label_index]
    return f"class_{label_index}"

# ==========================================
# 0. MASTER THESIS CONFIGURATION
# ==========================================
# Change these configuration values to select the experiment.
CURRENT_DATASET = 'XSUB120'
CURRENT_ENSEMBLE = '4-stream'     # Options: 'pure', '2-stream', '3-stream', '4-stream'
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
GIF_DIR = RESULT_DIR / "xai_gifs"
FRAME_DIR = RESULT_DIR / "xai_frames"
MANIFEST_PATH = RESULT_DIR / "explanation_manifest.json"

for output_dir in (NPY_DIR, GIF_DIR, FRAME_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. LOAD THE COMMITTEE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(f"Loading {CURRENT_ENSEMBLE} Architecture for {CURRENT_DATASET} on {device}...")

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
    
    gcn.eval()
    trans.eval()
    cls.eval()
    return gcn, trans, cls

# Only load the models that are actually active to save GPU RAM!
experts = {}
if 'JBV' in ACTIVE_STREAMS: experts['JBV'] = load_expert(9, 'jbv')
if 'B' in ACTIVE_STREAMS:   experts['B']   = load_expert(3, 'bones')
if 'J' in ACTIVE_STREAMS:   experts['J']   = load_expert(3, 'pure_joints')
if 'V' in ACTIVE_STREAMS:   experts['V']   = load_expert(3, 'pure_velocity')

# ==========================================
# 2. BATCH PROCESSING LOOP
# ==========================================
print(f"Loading Test Dataset for {CURRENT_DATASET}...")
dataset = NTUSkeletonDataset(
    data_folder=str(dataset_split_path(DATASET_NAME, 'test')),
    max_frames=100,
)

kinect_bones = [
    (0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), 
    (8, 20), (9, 8), (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), 
    (15, 14), (16, 0), (17, 16), (18, 17), (19, 18), (21, 22), 
    (22, 7), (23, 24), (24, 11)
]

parser = argparse.ArgumentParser(description='Batch process Multi-Pipe XAI heatmaps.')
parser.add_argument('--start', type=int, default=0, help='Start index for processing files.')
parser.add_argument('--end', type=int, default=None, help='End index for processing files.')
args = parser.parse_args()
end_idx = args.end if args.end is not None else len(dataset)
manifest_samples = {}
if MANIFEST_PATH.exists():
    with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        previous_manifest = json.load(manifest_file)
    manifest_samples = {
        sample["sample_id"]: sample
        for sample in previous_manifest.get("samples", [])
        if "sample_id" in sample
    }

for file_idx in tqdm(range(args.start, end_idx), desc="Processing XAI"):
    target_base = dataset.file_list[file_idx].replace('.pt', '')
    npy_path = NPY_DIR / f"{target_base}_fused.npy"
    gif_path = GIF_DIR / f"{target_base}_fused.gif"
    png_path = FRAME_DIR / f"{target_base}_fused_peak.png"
    
    if npy_path.exists() and gif_path.exists() and png_path.exists():
        existing_sample = manifest_samples.get(target_base, {})
        manifest_samples[target_base] = {
            **existing_sample,
            "sample_id": target_base,
            "heatmap": str(npy_path.relative_to(RESULT_DIR)),
            "gif": str(gif_path.relative_to(RESULT_DIR)),
            "static_image": str(png_path.relative_to(RESULT_DIR)),
        }
        continue

    # --- A. RUN MULTI-PIPE INFERENCE ---
    engineered_tensor, body_mask, true_label = dataset[file_idx]
    tensor_input = engineered_tensor.unsqueeze(0).to(device)
    body_mask = body_mask.unsqueeze(0).to(device)
    
    B, M, T, V, C = tensor_input.shape
    
    slices = {
        'JBV': select_pipeline_input(tensor_input, 'jbv'),
        'J': select_pipeline_input(tensor_input, 'joints'),
        'B': select_pipeline_input(tensor_input, 'bones'),
        'V': select_pipeline_input(tensor_input, 'velocity'),
    }
    
    differential_heatmaps = {}
    fused_probs = torch.zeros(1, NUM_CLASSES).to(device)
    
    with torch.no_grad():
        # DYNAMIC LOOP: Only runs the streams you activated
        for name in ACTIVE_STREAMS:
            gcn, trans, cls = experts[name]

            inp = slices[name].reshape(B * M, T, V, slices[name].shape[-1])
            feat = gcn(inp)
            frames = feat.shape[1]
            
            t_inp = torch.cat([feat, trans.global_node.expand(B*M, frames, 1, 128)], dim=2)

            video_rep, attention_matrix = trans(t_inp, B, M, body_mask=body_mask, return_attention=True)
            
            logits = cls(video_rep)
            probs = torch.softmax(logits, dim=1)
            
            fused_probs += WEIGHTS[name] * probs 
            
            raw_attn = attention_matrix[0].cpu().numpy() 
            diff_attn = apply_differential_xai(raw_attn)
            differential_heatmaps[name] = diff_attn

    pred_prob, predicted_class = torch.max(fused_probs, 1)
    pred_label = predicted_class.item()
    pred_confidence = pred_prob.item() * 100

    # --- B. FUSE THE COMMITTEE'S BRAINS ---
    # DYNAMIC CANVAS: Automatically sizes itself to the first active stream
    first_active_stream = ACTIVE_STREAMS[0]
    fused_matrix = np.zeros_like(differential_heatmaps[first_active_stream])
    
    # DYNAMIC FUSION: Only fuses what is active
    for name in ACTIVE_STREAMS:
        fused_matrix += WEIGHTS[name] * differential_heatmaps[name]
        
    final_heat_scores = normalize_global_heatmap(fused_matrix)
    np.save(npy_path, final_heat_scores)

    # --- C. FIND THE ABSOLUTE PEAK FRAME ---
    peak_body, peak_frame, true_max_joint = np.unravel_index(np.argmax(final_heat_scores), final_heat_scores.shape)

    # --- D. RENDER THE MASTER ENSEMBLE GIF ---
    skeleton_filename = target_base + '.skeleton'
    raw_file_path = dataset_split_path(DATASET_NAME, 'test') / 'raw_text' / skeleton_filename
    raw_skeleton_tensor = dataset.parse_single_skeleton(raw_file_path)
    
    if raw_skeleton_tensor is None:
        continue
        
    actual_frames = raw_skeleton_tensor.shape[0]
    
    if actual_frames > 100:
        indices = np.linspace(0, 99, actual_frames).astype(int)
        heat_scores = final_heat_scores[:, indices, :]
    else:
        heat_scores = final_heat_scores[:, :actual_frames, :]
        
    all_coords = raw_skeleton_tensor[:, :, :, 0:3]
    valid_coords = all_coords[np.any(all_coords != 0, axis=-1)]
    if len(valid_coords) == 0: continue
        
    mid_x, mid_y, mid_z = np.mean(valid_coords[:, 0]), np.mean(valid_coords[:, 1]), np.mean(valid_coords[:, 2])

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    cmap = plt.get_cmap('coolwarm') 

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
    cbar.ax.set_title('Fused Kinematic Delta', fontsize=10, pad=10)

    def update(frame_idx):
            ax.clear()
            ax.set_xlim(mid_x - 1, mid_x + 1)
            ax.set_ylim(mid_z - 1, mid_z + 1) 
            ax.set_zlim(mid_y - 1, mid_y + 1) 
            
            true_action_name = action_name(true_label.item())
            
            if pred_label == true_label.item():
                status = f"Correct ({pred_confidence:.1f}%)"
            else:
                status = f"Wrong: {action_name(pred_label)} ({pred_confidence:.1f}%)"
                
            # DYNAMIC TITLE: Now says "4-stream Ensemble XAI" or "pure Ensemble XAI"
            ax.set_title(f'Action: {true_action_name}\nStatus: {status} | {CURRENT_ENSEMBLE.upper()} Ensemble XAI', color='black', pad=10)
            
            for body_idx in range(2):
                current_skeleton = all_coords[frame_idx, body_idx]
                current_heat = heat_scores[body_idx, frame_idx]
                
                valid_joints = np.any(current_skeleton != 0, axis=-1)
                if not np.any(valid_joints): continue
                
                xs, ys, zs = current_skeleton[:, 0], current_skeleton[:, 1], current_skeleton[:, 2]
                colors = cmap(current_heat)
                
                ax.scatter(xs[valid_joints], zs[valid_joints], ys[valid_joints], 
                        c=colors[valid_joints], s=60, edgecolors='black', linewidth=1, zorder=2)
                
                for bone in kinect_bones:
                    j1, j2 = bone
                    if valid_joints[j1] and valid_joints[j2]:
                        ax.plot([xs[j1], xs[j2]], [zs[j1], zs[j2]], [ys[j1], ys[j2]], c='black', linewidth=2, zorder=1)

    if actual_frames > 100:
        actual_peak_frame = np.searchsorted(indices, peak_frame)
    else:
        actual_peak_frame = peak_frame
        
    update(actual_peak_frame)
    plt.savefig(png_path, bbox_inches='tight', dpi=300, facecolor='white')

    ani = animation.FuncAnimation(fig, update, frames=actual_frames, interval=50)
    ani.save(gif_path, writer='pillow', fps=20)
    plt.close(fig)
    manifest_samples[target_base] = {
        "sample_id": target_base,
        "true_label": int(true_label.item()),
        "true_action": action_name(int(true_label.item())),
        "predicted_label": pred_label,
        "predicted_action": action_name(pred_label),
        "predicted_confidence": pred_confidence / 100,
        "heatmap": str(npy_path.relative_to(RESULT_DIR)),
        "gif": str(gif_path.relative_to(RESULT_DIR)),
        "static_image": str(png_path.relative_to(RESULT_DIR)),
    }

manifest = {
    "schema_version": 1,
    "dataset": DATASET_NAME,
    "run_id": RUN_ID,
    "ensemble": CURRENT_ENSEMBLE,
    "streams": WEIGHTS,
    "samples": [
        manifest_samples[sample_id]
        for sample_id in sorted(manifest_samples)
    ],
}
with MANIFEST_PATH.open("w", encoding="utf-8") as manifest_file:
    json.dump(manifest, manifest_file, indent=2, sort_keys=True)

print(f"XAI explanation generation complete for {DATASET_NAME} ({CURRENT_ENSEMBLE}).")