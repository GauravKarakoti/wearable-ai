"""
params.py -- exact parameter count for the Grand Challenge submission
form. Sums the frozen V-JEPA2 encoder, the frozen Qwen2-VL-2B decoder, and
the trained bridge -- all three together are what actually run at inference
time, so all three count toward your submission's total.
"""

import torch
from transformers import AutoModel, Qwen2VLForConditionalGeneration

JEPA_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
QWEN_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print("Loading V-JEPA2 encoder...")
    jepa = AutoModel.from_pretrained(JEPA_MODEL_ID)
    jepa_params = count_params(jepa)
    del jepa

    print("Loading Qwen2-VL-2B-Instruct...")
    # FIX: Use the specific Qwen2-VL generation class
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, 
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True
    )
    qwen_params = count_params(qwen)
    del qwen

    bridge = torch.nn.Sequential(
        torch.nn.Linear(1024, 1536),
        torch.nn.GELU(),
        torch.nn.Linear(1536, 1536),
    )
    bridge_params = count_params(bridge)

    total = jepa_params + qwen_params + bridge_params

    def fmt(n):
        return f"{n:,}  ({n / 1e9:.4f}B)"

    print("\n--- EXACT PARAMETER COUNTS ---")
    print(f"V-JEPA2 encoder (frozen):  {fmt(jepa_params)}")
    print(f"Qwen2-VL decoder (frozen): {fmt(qwen_params)}")
    print(f"MLP bridge (trained):      {fmt(bridge_params)}")
    print(f"TOTAL:                     {fmt(total)}")
    print(f"\nBridge as % of total: {bridge_params / total * 100:.4f}%")
    print(f"\nFor the submission form:")
    print(f"  Total params (billions):  {total / 1e9:.4f}")
    print(f"  Active params (billions): {total / 1e9:.4f}  (same as total -- dense architecture, no MoE/routing)")


if __name__ == "__main__":
    main()