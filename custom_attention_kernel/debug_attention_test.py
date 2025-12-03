"""
Simple debug test to isolate the attention kernel issue.
Tests with known small inputs to verify correctness.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.cpp_extension import load


def test_simple_case():
    """Test with very small, easy-to-verify inputs."""
    print("=" * 80)
    print("SIMPLE DEBUG TEST")
    print("=" * 80)
    
    device = torch.device('cuda')
    
    # Small test case
    batch = 2
    num_heads = 2
    tokens = 8
    head_dim = 8
    
    print(f"\nTest configuration:")
    print(f"  Batch: {batch}")
    print(f"  Heads: {num_heads}")
    print(f"  Tokens: {tokens}")
    print(f"  Head dim: {head_dim}")
    
    # Create simple test data
    torch.manual_seed(42)
    q = torch.randn(batch, num_heads, tokens, head_dim, device=device)
    k = torch.randn(batch, num_heads, tokens, head_dim, device=device)
    v = torch.randn(batch, num_heads, tokens, head_dim, device=device)
    
    scale = head_dim ** -0.5
    q = q * scale
    
    # PyTorch reference
    print("\nRunning PyTorch reference...")
    softmax = nn.Softmax(dim=-1)
    attn_scores = q @ k.transpose(-2, -1)
    attn = softmax(attn_scores)
    output_pytorch = attn @ v
    
    print(f"  Attention scores range: [{attn_scores.min():.4f}, {attn_scores.max():.4f}]")
    print(f"  Attention probs range: [{attn.min():.4f}, {attn.max():.4f}]")
    print(f"  Output range: [{output_pytorch.min():.4f}, {output_pytorch.max():.4f}]")
    
    # Compile and run CUDA kernel
    print("\nCompiling CUDA kernel...")
    cuda_dir = Path(__file__).parent
    naive_attention_module = load(
        name="naive_attention",
        sources=[
            str(cuda_dir / "kernels/naive_attention_forward.cpp"),
            str(cuda_dir / "kernels/naive_attention_forward.cu")
        ],
        verbose=False
    )
    
    print("Running CUDA kernel...")
    output_cuda = naive_attention_module.naive_attention_forward(
        q, k, v, batch, num_heads, tokens, head_dim
    )
    torch.cuda.synchronize()
    
    print(f"  CUDA output range: [{output_cuda.min():.4f}, {output_cuda.max():.4f}]")
    
    # Compare
    print("\nComparing outputs...")
    output_pytorch_cpu = output_pytorch.cpu().numpy()
    output_cuda_cpu = output_cuda.cpu().numpy()
    
    max_diff = np.max(np.abs(output_pytorch_cpu - output_cuda_cpu))
    mean_diff = np.mean(np.abs(output_pytorch_cpu - output_cuda_cpu))
    
    print(f"  Max difference: {max_diff:.9f}")
    print(f"  Mean difference: {mean_diff:.9f}")
    
    if max_diff < 1e-5:
        print("  ✅ PASS - Outputs match!")
    else:
        print("  ❌ FAIL - Outputs differ!")
        print(f"\n  PyTorch[0,0,0,:]: {output_pytorch_cpu[0,0,0,:]}")
        print(f"  CUDA[0,0,0,:]:    {output_cuda_cpu[0,0,0,:]}")
        print(f"  Diff:             {output_pytorch_cpu[0,0,0,:] - output_cuda_cpu[0,0,0,:]}")


def test_large_tokens():
    """Test with realistic token count (343)."""
    print("\n" + "=" * 80)
    print("LARGE TOKEN TEST (343 tokens)")
    print("=" * 80)
    
    device = torch.device('cuda')
    
    # Realistic Swin UNETR dimensions
    batch = 4  # Small batch
    num_heads = 3
    tokens = 343  # 7x7x7 window
    head_dim = 16  # feature_size=48 / num_heads=3
    
    print(f"\nTest configuration:")
    print(f"  Batch: {batch}")
    print(f"  Heads: {num_heads}")
    print(f"  Tokens: {tokens}")
    print(f"  Head dim: {head_dim}")
    
    # Create test data
    torch.manual_seed(123)
    q = torch.randn(batch, num_heads, tokens, head_dim, device=device)
    k = torch.randn(batch, num_heads, tokens, head_dim, device=device)
    v = torch.randn(batch, num_heads, tokens, head_dim, device=device)
    
    scale = head_dim ** -0.5
    q = q * scale
    
    # PyTorch reference
    print("\nRunning PyTorch reference...")
    softmax = nn.Softmax(dim=-1)
    attn_scores = q @ k.transpose(-2, -1)
    attn = softmax(attn_scores)
    output_pytorch = attn @ v
    
    print(f"  Attention scores range: [{attn_scores.min():.4f}, {attn_scores.max():.4f}]")
    print(f"  Attention probs sum (should be 1.0): {attn[0,0,0,:].sum():.6f}")
    print(f"  Output range: [{output_pytorch.min():.4f}, {output_pytorch.max():.4f}]")
    
    # Compile and run CUDA kernel
    print("\nCompiling CUDA kernel...")
    cuda_dir = Path(__file__).parent
    naive_attention_module = load(
        name="naive_attention",
        sources=[
            str(cuda_dir / "kernels/naive_attention_forward.cpp"),
            str(cuda_dir / "kernels/naive_attention_forward.cu")
        ],
        verbose=False
    )
    
    print("Running CUDA kernel...")
    output_cuda = naive_attention_module.naive_attention_forward(
        q, k, v, batch, num_heads, tokens, head_dim
    )
    torch.cuda.synchronize()
    
    print(f"  CUDA output range: [{output_cuda.min():.4f}, {output_cuda.max():.4f}]")
    
    # Compare
    print("\nComparing outputs...")
    output_pytorch_cpu = output_pytorch.cpu().numpy()
    output_cuda_cpu = output_cuda.cpu().numpy()
    
    max_diff = np.max(np.abs(output_pytorch_cpu - output_cuda_cpu))
    mean_diff = np.mean(np.abs(output_pytorch_cpu - output_cuda_cpu))
    relative_diff = max_diff / (np.abs(output_pytorch_cpu).mean() + 1e-8)
    
    print(f"  Max absolute difference: {max_diff:.9f}")
    print(f"  Mean absolute difference: {mean_diff:.9f}")
    print(f"  Relative difference: {relative_diff:.6f}")
    
    # Check for NaN/Inf
    print(f"\n  Checking for NaN/Inf:")
    print(f"    PyTorch has NaN: {torch.isnan(output_pytorch).any().item()}")
    print(f"    PyTorch has Inf: {torch.isinf(output_pytorch).any().item()}")
    print(f"    CUDA has NaN: {torch.isnan(output_cuda).any().item()}")
    print(f"    CUDA has Inf: {torch.isinf(output_cuda).any().item()}")
    
    tolerance = 1e-4
    if max_diff < tolerance:
        print(f"  ✅ PASS - Outputs match within tolerance ({tolerance})!")
    else:
        print(f"  ❌ FAIL - Outputs differ beyond tolerance ({tolerance})")
        print(f"\n  First few values comparison:")
        print(f"  PyTorch[0,0,0,:5]: {output_pytorch_cpu[0,0,0,:5]}")
        print(f"  CUDA[0,0,0,:5]:    {output_cuda_cpu[0,0,0,:5]}")
        print(f"  Diff:              {output_pytorch_cpu[0,0,0,:5] - output_cuda_cpu[0,0,0,:5]}")


if __name__ == "__main__":
    test_simple_case()
    test_large_tokens()
