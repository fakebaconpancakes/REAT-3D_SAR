import torch
import torch.nn as nn
import time

try:
    from thop import profile, clever_format
except ImportError:
    profile = None
    clever_format = None

from models.spatial_gcn import Spatial_GCN_Layer
from models.temporal_brain import Temporal_Brain_Layer


class REAT_Model(nn.Module):
    def __init__(self, in_channels=9, num_classes=120):
        super().__init__()
        self.gcn = Spatial_GCN_Layer(in_channels=in_channels, out_channels=128)
        self.transformer = Temporal_Brain_Layer(embed_dim=128, num_heads=4, max_frames=100, max_bodies=2)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        B, M, T, V, C = x.shape
        gcn_input = x.reshape(B * M, T, V, C)
        gcn_features = self.gcn(gcn_input)
        
        frames = gcn_features.shape[1]
        global_node_expanded = self.transformer.global_node.expand(B * M, frames, 1, 128)
        transformer_input = torch.cat([gcn_features, global_node_expanded], dim=2)
        
        return self.classifier(self.transformer(transformer_input, B, M))


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _linear_macs(num_tokens, in_dim, out_dim):
    return num_tokens * in_dim * out_dim


def _ffn_macs(num_tokens, embed_dim, expansion=4):
    hidden = embed_dim * expansion
    return _linear_macs(num_tokens, embed_dim, hidden) + _linear_macs(num_tokens, hidden, embed_dim)


def _spatial_mask_pairs(room_map, global_idx):
    room_map = room_map.tolist()
    total_pairs = 0
    seq_len = len(room_map)
    for q_idx in range(seq_len):
        for kv_idx in range(seq_len):
            is_global_q = q_idx == global_idx
            is_global_kv = kv_idx == global_idx
            same_room = q_idx < 25 and kv_idx < 25 and room_map[q_idx] == room_map[kv_idx]
            if is_global_q or is_global_kv or same_room:
                total_pairs += 1
    return total_pairs


def _temporal_window_pairs(num_bodies, num_frames, window_size=5):
    per_body_pair = sum(
        1
        for query_time in range(num_frames)
        for key_time in range(num_frames)
        if abs(query_time - key_time) <= window_size
    )
    body_pairs = num_bodies * num_bodies * per_body_pair
    video_token_pairs = 2 * num_bodies * num_frames + 1
    return body_pairs + video_token_pairs


def estimate_reat_macs(model, x_shape):
    B, M, T, V, C = x_shape
    E = model.transformer.embed_dim
    H = model.transformer.num_heads
    S = V + 1  
    d = E // H
    B_M = B * M

    # ---- Spatial GCN ----
    tokens_gcn = B_M * T * V
    macs_gcn_linear = _linear_macs(tokens_gcn, C, E)
    macs_gcn_einsum = B_M * T * V * E * V

    # ---- Temporal Brain / Spatial block ----
    tokens_spatial = B_M * T * S
    macs_spatial_qkv = _linear_macs(tokens_spatial, E, 3 * E)

    mask_pairs_per_frame_head = _spatial_mask_pairs(
        model.transformer.room_map, model.transformer.global_node_idx
    )
    macs_spatial_attn = B_M * T * H * mask_pairs_per_frame_head * (2 * d)

    macs_spatial_out = _linear_macs(tokens_spatial, E, E)
    macs_spatial_ffn = _ffn_macs(tokens_spatial, E, expansion=4)

    # ---- Temporal block ----
    L = (M * T) + 1  
    tokens_temporal = B * L 
    
    macs_temporal_qkv = _linear_macs(tokens_temporal, E, 3 * E)
    temporal_pairs = _temporal_window_pairs(M, T, window_size=5)
    macs_temporal_attn = B * H * temporal_pairs * (2 * d)
    
    macs_temporal_out = _linear_macs(tokens_temporal, E, E)
    macs_temporal_ffn = _ffn_macs(tokens_temporal, E, expansion=4)
    macs_classifier = _linear_macs(B, E, model.classifier.out_features)

    total_macs = (
        macs_gcn_linear
        + macs_gcn_einsum
        + macs_spatial_qkv
        + macs_spatial_attn
        + macs_spatial_out
        + macs_spatial_ffn
        + macs_temporal_qkv
        + macs_temporal_attn
        + macs_temporal_out
        + macs_temporal_ffn
        + macs_classifier
    )

    return {
        "total": total_macs,
        "breakdown": {
            "gcn_linear": macs_gcn_linear,
            "gcn_einsum": macs_gcn_einsum,
            "spatial_qkv_proj": macs_spatial_qkv,
            "spatial_attention_sparse": macs_spatial_attn,
            "spatial_out_proj": macs_spatial_out,
            "spatial_ffn": macs_spatial_ffn,
            "temporal_qkv_proj": macs_temporal_qkv,
            "temporal_attention_windowed": macs_temporal_attn,
            "temporal_out_proj": macs_temporal_out,
            "temporal_ffn": macs_temporal_ffn,
            "classification_head": macs_classifier,
        },
    }


def benchmark_inference(model, input_tensor, device, warmup=5, repeats=20):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(input_tensor)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            model(input_tensor)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - start)
    latency_ms = 1000.0 * sum(durations) / len(durations)
    return {
        "mean_latency_ms": latency_ms,
        "samples_per_second": input_tensor.shape[0] * 1000.0 / latency_ms,
        "cuda_peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda" else None
        ),
        "warmup_iterations": warmup,
        "measurement_iterations": repeats,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building Multi-Stream REAT Profiler on {device}...\n")

    # 1. Profile the 9-Channel JBV Expert
    model_9ch = REAT_Model(in_channels=9).to(device)
    macs_9ch = estimate_reat_macs(model_9ch, (1, 2, 100, 25, 9))["total"]
    params_9ch = count_trainable_params(model_9ch)

    # 2. Profile the 3-Channel Pure Expert (Bones, Joints, Velocity)
    model_3ch = REAT_Model(in_channels=3).to(device)
    macs_3ch = estimate_reat_macs(model_3ch, (1, 2, 100, 25, 3))["total"]
    params_3ch = count_trainable_params(model_3ch)

    # 3. Define the Ensemble Compositions
    ensembles = {
        "JBV":               {"9ch": 1, "3ch": 0},
        "JBV : V":           {"9ch": 1, "3ch": 1},
        "JBV : J":           {"9ch": 1, "3ch": 1},
        "JBV : J : V":       {"9ch": 1, "3ch": 2},
        "JBV : B : V":       {"9ch": 1, "3ch": 2},
        "JBV : B : J : V":   {"9ch": 1, "3ch": 3},
    }

    print("=" * 70)
    print("  REAT ENSEMBLE COMPLEXITY REPORT (Analytic) ")
    print("=" * 70)
    print(f"{'Ensemble Type':<25} | {'Total Params':<15} | {'Analytic MACs'}")
    print("-" * 70)

    # 4. Calculate and display aggregated totals
    for name, config in ensembles.items():
        total_params = (config["9ch"] * params_9ch) + (config["3ch"] * params_3ch)
        total_macs = (config["9ch"] * macs_9ch) + (config["3ch"] * macs_3ch)
        print(f"{name:<25} | {total_params:<15,} | {total_macs:,}")
    
    print("=" * 70)

    # THOP cross-check for just the 9-channel base model to ensure alignment
    if profile is not None and clever_format is not None:
        print("\nCalculating THOP result for 9-Channel Base Model...")
        dummy_input = torch.randn(1, 2, 100, 25, 9).to(device)
        thop_macs, thop_params = profile(model_9ch, inputs=(dummy_input,), verbose=False)
        thop_macs_str, thop_params_str = clever_format([thop_macs, thop_params], "%.3f")
        delta = macs_9ch - thop_macs
        delta_pct = (delta / macs_9ch) * 100 if macs_9ch else 0.0

        print("\n" + "-" * 70)
        print(f"THOP Parameters (9ch):      {thop_params_str}")
        print(f"THOP MACs (9ch):            {thop_macs_str}")
        print(f"Analytic vs THOP Discrepancy: {int(delta):,} ({delta_pct:.2f}%)")
        print("-" * 70)

    print("\nAnalytic MAC estimates count allowed masked attention pairs and include the classifier.")
    print("They exclude bias, normalization, softmax, activation, and data movement; 2x MAC is only a FLOP convention.")
    print("Runtime kernels may execute padding or sparse blocks differently, so this is not a hardware operation measurement.")
    print("GPU timing and peak allocation are measured separately:")
    benchmark_input = torch.randn(1, 2, 100, 25, 9, device=device)
    benchmark = benchmark_inference(model_9ch, benchmark_input, device)
    print(f"Mean latency: {benchmark['mean_latency_ms']:.3f} ms/sample batch")
    print(f"Throughput: {benchmark['samples_per_second']:.2f} samples/s")
    if benchmark["cuda_peak_allocated_mb"] is not None:
        print(f"CUDA peak allocated memory: {benchmark['cuda_peak_allocated_mb']:.2f} MiB")
    else:
        print("CUDA peak memory: unavailable on CPU; run this profiler on CUDA for GPU memory figures.")


if __name__ == "__main__":
    main()
