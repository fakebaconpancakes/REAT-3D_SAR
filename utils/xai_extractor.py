import numpy as np

def apply_differential_xai(raw_attention_matrix, epsilon=1e-6):
    """
    Applies the Differential Fold Change filter to the raw attention matrix.
    Expected Input Shape: (Bodies, Time, 25)
    """
    M, T, V = raw_attention_matrix.shape
    differential_matrix = np.zeros_like(raw_attention_matrix)
    
    # Process each body independently
    for m in range(M):
        body_probs = raw_attention_matrix[m] # Shape: (Time, 25)
        
        # 1. Calculate Shannon Entropy per frame: -sum(p * log(p))
        # High entropy = attention is spread out (The AI is "resting")
        # Low entropy = attention is focused (The AI is tracking an action)
        entropies = -np.sum(body_probs * np.log(body_probs + 1e-9), axis=1)
        
        # 2. Identify the Resting Frame (Max Entropy)
        resting_frame_idx = np.argmax(entropies)
        resting_array = body_probs[resting_frame_idx, :] # Shape: (25,)
        
        # 3. Relative Fold Change Filter: max(0, (Peak - Rest) / Rest)
        # We broadcast the 1D resting_array across all frames
        diff = np.maximum((body_probs - resting_array[np.newaxis, :]) / (resting_array[np.newaxis, :] + epsilon), 0)
        
        differential_matrix[m] = diff
        
    return differential_matrix

def normalize_global_heatmap(fused_matrix):
    """
    Normalizes the final fused matrix to a 0.0 - 1.0 scale globally,
    so that only the absolute peak action of the video glows bright red.
    """
    max_val = np.max(fused_matrix)
    if max_val > 0:
        return fused_matrix / max_val
    return fused_matrix