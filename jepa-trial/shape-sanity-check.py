"""
Standalone shape sanity check -- mock tensors only, no real models/video needed.
"""

import torch
import torch.nn as nn

JEPA_HIDDEN = 1024

# Swap this to match whichever text backbone you actually decide to use:
#   Qwen2-VL-2B-Instruct   -> 1536
#   Qwen2.5-1.5B-Instruct  -> 1536
#   Llama-3.2-1B-Instruct  -> 2048
LLM_HIDDEN = 1536

vjepa_tensor = torch.randn(1, 8192, JEPA_HIDDEN)
mock_text_tokens = torch.randn(1, 20, LLM_HIDDEN)

projection_layer = nn.Linear(JEPA_HIDDEN, LLM_HIDDEN)

projected_video_tokens = projection_layer(vjepa_tensor)
print("Projected video tokens shape:", tuple(projected_video_tokens.shape))
assert projected_video_tokens.shape == (1, 8192, LLM_HIDDEN), "Projection shape mismatch!"

# FIX: dim=1 (sequence dimension), not dim=0 (batch dimension).
combined_embeddings = torch.cat([projected_video_tokens, mock_text_tokens], dim=1)
print("Combined embeddings shape:", tuple(combined_embeddings.shape))
assert combined_embeddings.shape == (1, 8192 + 20, LLM_HIDDEN), "Concat shape mismatch!"

print("\nShapes align. This confirms the CONCEPT only -- it doesn't test whether")
print("the real V-JEPA2 output size matches 8192 tokens for your actual video")
print("length/resolution, and it doesn't test anything about training dynamics.")
print("Our other scripts (jepa_one_video_trial.py, jepa_qwen_quick_trained_bridge.py)")
print("already do the real end-to-end version of this with actual model weights.")