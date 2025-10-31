import torch
import numpy as np
from monai.networks.nets.swin_unetr import WindowAttention
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cuda.fused_qkv import fused_qkv_projection

def test_qkv_projection():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define test parameters
    batch_size = 4
    seq_length = 49  # 7x7 window size
    dim = 96        # Feature dimension
    num_heads = 3
    
    # Create dummy input tensor
    x = torch.randn(batch_size, seq_length, dim, device='cuda')
    
    # Create MONAI WindowAttention instance
    window_size = (7, 7)  # Standard window size in Swin UNETR
    monai_attn = WindowAttention(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
        qkv_bias=True
    ).cuda()
    
    # Get MONAI QKV projection output
    with torch.no_grad():
        monai_qkv = monai_attn.qkv(x).reshape(batch_size, seq_length, 3, num_heads, dim // num_heads)
        monai_qkv = monai_qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
    
    # Get CUDA fused QKV projection output
    with torch.no_grad():
        cuda_qkv = fused_qkv_projection(
            x,
            monai_attn.qkv.weight,
            monai_attn.qkv.bias,
            num_heads
        )
    
    # Compare outputs
    max_diff = torch.max(torch.abs(monai_qkv - cuda_qkv))
    print(f"Maximum absolute difference between MONAI and CUDA outputs: {max_diff}")
    
    # Save outputs for detailed comparison if needed
    torch.save({
        'monai_qkv': monai_qkv.cpu(),
        'cuda_qkv': cuda_qkv.cpu(),
        'input': x.cpu(),
        'weight': monai_attn.qkv.weight.cpu(),
        'bias': monai_attn.qkv.bias.cpu()
    }, 'models/custom_swin_unetr/tests/results/qkv_test_outputs.pt')

    # Check if outputs are close enough
    assert torch.allclose(monai_qkv, cuda_qkv, rtol=1e-3, atol=1e-3), \
        "MONAI and CUDA QKV projection outputs do not match!"
    
    print("✓ MONAI and CUDA QKV projection outputs match within tolerance")

if __name__ == "__main__":
    test_qkv_projection()