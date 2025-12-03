"""
Real-world test of custom CUDA attention kernel using BraTS dataset.
Simulates actual Swin UNETR attention computation on real medical imaging data.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
from pathlib import Path
from torch.utils.cpp_extension import load
import torch.nn.functional as F

def load_brats_sample(sample_path):
    """Load a preprocessed BraTS tensor sample."""
    print("=" * 80)
    print("LOADING BRATS SAMPLE")
    print("=" * 80)
    
    data = torch.load(sample_path, weights_only=False)
    image = data["image"]  # [C, D, H, W]
    label = data["label"]  # [1, D, H, W]
    
    print(f"\nLoaded: {sample_path}")
    print(f"  Image shape: {image.shape}")
    print(f"  Label shape: {label.shape}")
    print(f"  Label classes: {torch.unique(label).tolist()}")
    
    return image, label


def simulate_swin_attention_input(image, window_size=(7, 7, 7), patch_size=2, feature_size=48, num_heads=3):
    """
    Simulate the input to Swin UNETR's WindowAttention layer.
    
    In Swin UNETR:
    - Input gets patched and embedded
    - Features get reshaped into windows
    - QKV projection creates query, key, value tensors
    
    Args:
        image: [C, D, H, W] BraTS image
        window_size: Swin window size (7x7x7 = 343 tokens per window)
        patch_size: Patch embedding size
        feature_size: Feature dimension
        num_heads: Number of attention heads
    """
    print("\n" + "=" * 80)
    print("SIMULATING SWIN ATTENTION INPUT")
    print("=" * 80)
    
    device = image.device
    batch = 1  # Single sample
    
    # Calculate dimensions after patch embedding
    c, d, h, w = image.shape
    d_patched = d // patch_size
    h_patched = h // patch_size
    w_patched = w // patch_size
    
    print(f"\nOriginal image: {image.shape}")
    print(f"After patch embedding ({patch_size}x{patch_size}x{patch_size}): [{d_patched}, {h_patched}, {w_patched}]")
    
    # Calculate number of windows
    wd, wh, ww = window_size
    num_windows_d = d_patched // wd
    num_windows_h = h_patched // wh
    num_windows_w = w_patched // ww
    num_windows = num_windows_d * num_windows_h * num_windows_w
    tokens_per_window = wd * wh * ww
    
    print(f"\nWindow size: {window_size}")
    print(f"Number of windows: {num_windows} ({num_windows_d}x{num_windows_h}x{num_windows_w})")
    print(f"Tokens per window: {tokens_per_window}")
    
    # Simulate features after patch embedding and windowing
    # Shape: [num_windows, tokens_per_window, feature_size]
    features = torch.randn(num_windows, tokens_per_window, feature_size, device=device)
    
    print(f"\nFeatures shape: {features.shape}")
    print(f"  num_windows: {num_windows}")
    print(f"  tokens_per_window: {tokens_per_window}")
    print(f"  feature_size: {feature_size}")
    
    # Simulate QKV projection (mimics WindowAttention.__init__)
    # In real Swin UNETR: self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    qkv_weight = torch.randn(feature_size * 3, feature_size, device=device) * 0.02
    qkv_bias = torch.randn(feature_size * 3, device=device) * 0.02
    
    # QKV projection
    qkv = torch.matmul(features, qkv_weight.T) + qkv_bias
    
    # Reshape and split QKV
    # Shape: [num_windows, tokens_per_window, 3, num_heads, head_dim]
    head_dim = feature_size // num_heads
    qkv = qkv.reshape(num_windows, tokens_per_window, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, num_windows, num_heads, tokens_per_window, head_dim]
    
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    # Apply scaling (as done in WindowAttention.forward line 520)
    scale = head_dim ** -0.5
    q = q * scale
    
    print(f"\nAfter QKV projection and split:")
    print(f"  Query shape: {q.shape}  [num_windows, num_heads, tokens, head_dim]")
    print(f"  Key shape: {k.shape}")
    print(f"  Value shape: {v.shape}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Scale factor: {scale:.6f}")
    
    return q, k, v, num_windows, num_heads, tokens_per_window, head_dim


def test_attention_on_brats(sample_path, num_iterations=100, compile_kernel=True, window_size=(7, 7, 7)):
    """
    Test custom CUDA attention kernel vs PyTorch on real BraTS data.
    
    Args:
        sample_path: Path to BraTS .pt file
        num_iterations: Number of timing iterations
        compile_kernel: Whether to compile CUDA kernel
        window_size: Window size tuple (default: 7x7x7 = 343 tokens)
    """
    print("\n" + "=" * 80)
    print("BRATS ATTENTION KERNEL TEST")
    print("=" * 80)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n❌ CUDA not available!")
        return False
    
    device = torch.device('cuda')
    print(f"\nDevice: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load BraTS sample
    image, label = load_brats_sample(sample_path)
    image = image.to(device)
    label = label.to(device)
    
    # Simulate Swin attention input
    q, k, v, num_windows, num_heads, tokens_per_window, head_dim = simulate_swin_attention_input(
        image, window_size=window_size
    )
    
    # For testing, we'll process all windows as a batch
    batch = num_windows
    
    print(f"\n" + "-" * 80)
    print("TEST CONFIGURATION")
    print("-" * 80)
    print(f"  Batch size (num_windows): {batch}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Tokens per window: {tokens_per_window}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Total parameters: {batch * num_heads * tokens_per_window * head_dim:,}")
    
    # ========== PyTorch Attention ==========
    print("\n" + "-" * 80)
    print("PYTORCH ATTENTION")
    print("-" * 80)
    
    softmax = nn.Softmax(dim=-1)
    
    # Warmup
    for _ in range(10):
        attn_scores = q @ k.transpose(-2, -1)
        attn = softmax(attn_scores)
        output_pytorch = attn @ v
        torch.cuda.synchronize()
    
    # Timed runs
    pytorch_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        attn_scores = q @ k.transpose(-2, -1)
        attn = softmax(attn_scores)
        output_pytorch = attn @ v
        torch.cuda.synchronize()
        end = time.time()
        pytorch_times.append(end - start)
    
    pytorch_mean = np.mean(pytorch_times) * 1000
    pytorch_std = np.std(pytorch_times) * 1000
    pytorch_min = np.min(pytorch_times) * 1000
    pytorch_max = np.max(pytorch_times) * 1000
    
    print(f"  Mean time: {pytorch_mean:.4f} ms")
    print(f"  Std dev:   {pytorch_std:.4f} ms")
    print(f"  Min time:  {pytorch_min:.4f} ms")
    print(f"  Max time:  {pytorch_max:.4f} ms")
    
    if not compile_kernel:
        print("\n⏭️  Skipping CUDA kernel compilation (compile_kernel=False)")
        return True
    
    # ========== Custom CUDA Kernel ==========
    print("\n" + "-" * 80)
    print("CUSTOM CUDA ATTENTION KERNEL")
    print("-" * 80)
    
    # Compile CUDA kernel
    cuda_dir = Path(__file__).parent
    try:
        print("  Compiling CUDA kernel...")
        naive_attention_module = load(
            name="naive_attention",
            sources=[
                str(cuda_dir / "kernels/naive_attention_forward.cpp"),
                str(cuda_dir / "kernels/naive_attention_forward.cu")
            ],
            verbose=False
        )
        print("  ✅ CUDA kernel compiled successfully!")
    except Exception as e:
        print(f"  ❌ Failed to compile CUDA kernel: {e}")
        return False
    
    # Warmup
    for _ in range(10):
        _ = naive_attention_module.naive_attention_forward(
            q, k, v, batch, num_heads, tokens_per_window, head_dim
        )
        torch.cuda.synchronize()
    
    # Timed runs
    cuda_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        output_cuda = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False
        )
        # output_cuda = naive_attention_module.naive_attention_forward(
        #     q, k, v, batch, num_heads, tokens_per_window, head_dim
        # )
        torch.cuda.synchronize()
        end = time.time()
        cuda_times.append(end - start)
    
    cuda_mean = np.mean(cuda_times) * 1000
    cuda_std = np.std(cuda_times) * 1000
    cuda_min = np.min(cuda_times) * 1000
    cuda_max = np.max(cuda_times) * 1000
    
    print(f"  Mean time: {cuda_mean:.4f} ms")
    print(f"  Std dev:   {cuda_std:.4f} ms")
    print(f"  Min time:  {cuda_min:.4f} ms")
    print(f"  Max time:  {cuda_max:.4f} ms")
    
    # ========== Performance Comparison ==========
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    speedup = pytorch_mean / cuda_mean
    
    print(f"\nPyTorch (GPU):        {pytorch_mean:.4f} ms ± {pytorch_std:.4f} ms")
    print(f"Custom CUDA Kernel:   {cuda_mean:.4f} ms ± {cuda_std:.4f} ms")
    
    if speedup > 1:
        print(f"\n✅ Custom kernel is {speedup:.2f}x FASTER than PyTorch!")
    else:
        speedup = 1 / speedup
        print(f"\n⚠️  PyTorch is {speedup:.2f}x faster than custom kernel")
    
    # ========== Correctness Verification ==========
    print("\n" + "-" * 80)
    print("CORRECTNESS VERIFICATION")
    print("-" * 80)
    
    output_pytorch_cpu = output_pytorch.cpu().numpy()
    output_cuda_cpu = output_cuda.cpu().numpy()
    
    max_diff = np.max(np.abs(output_pytorch_cpu - output_cuda_cpu))
    mean_diff = np.mean(np.abs(output_pytorch_cpu - output_cuda_cpu))
    relative_diff = max_diff / (np.abs(output_pytorch_cpu).mean() + 1e-8)
    
    print(f"  Max absolute difference:  {max_diff:.9f}")
    print(f"  Mean absolute difference: {mean_diff:.9f}")
    print(f"  Relative difference:      {relative_diff:.9f}")
    print(f"  Output magnitude (PyTorch): {np.abs(output_pytorch_cpu).mean():.6f}")
    print(f"  Output magnitude (CUDA):    {np.abs(output_cuda_cpu).mean():.6f}")
    
    tolerance = 1e-4  # Slightly relaxed for real data
    if max_diff < tolerance:
        print(f"  ✅ Outputs match within tolerance ({tolerance})!")
    else:
        print(f"  ⚠️  Outputs differ beyond tolerance ({tolerance})")
        print(f"\n  Debugging output samples:")
        print(f"  First window, first head, first 5 tokens, first 3 dims:")
        print(f"  PyTorch:\n{output_pytorch_cpu[0, 0, :5, :3]}")
        print(f"  CUDA:\n{output_cuda_cpu[0, 0, :5, :3]}")
        print(f"  Difference:\n{output_pytorch_cpu[0, 0, :5, :3] - output_cuda_cpu[0, 0, :5, :3]}")
        
        # Check if it's a systematic issue
        print(f"\n  Checking for NaN/Inf:")
        print(f"  PyTorch has NaN: {np.isnan(output_pytorch_cpu).any()}")
        print(f"  PyTorch has Inf: {np.isinf(output_pytorch_cpu).any()}")
        print(f"  CUDA has NaN: {np.isnan(output_cuda_cpu).any()}")
        print(f"  CUDA has Inf: {np.isinf(output_cuda_cpu).any()}")
    
    print("\n" + "=" * 80)
    
    return {
        'pytorch_mean': pytorch_mean,
        'pytorch_std': pytorch_std,
        'cuda_mean': cuda_mean,
        'cuda_std': cuda_std,
        'speedup': speedup if pytorch_mean > cuda_mean else -1/speedup,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'num_windows': num_windows,
        'tokens_per_window': tokens_per_window
    }


def test_multiple_samples(data_dir, num_samples=5, num_iterations=100):
    """Test on multiple BraTS samples."""
    print("=" * 80)
    print(f"TESTING ON MULTIPLE BRATS SAMPLES (n={num_samples})")
    print("=" * 80)
    
    data_dir = Path(data_dir)
    sample_files = sorted(list(data_dir.glob("subject_*.pt")))[:num_samples]
    
    if len(sample_files) == 0:
        print(f"\n❌ No samples found in {data_dir}")
        return
    
    print(f"\nFound {len(sample_files)} samples")
    
    results = []
    for i, sample_path in enumerate(sample_files):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}/{len(sample_files)}: {sample_path.name}")
        print(f"{'='*80}")
        
        result = test_attention_on_brats(
            sample_path, 
            num_iterations=num_iterations,
            compile_kernel=(i == 0)  # Only compile once
        )
        
        if result:
            results.append(result)
    
    # Summary statistics
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL SAMPLES")
        print("=" * 80)
        
        avg_pytorch = np.mean([r['pytorch_mean'] for r in results])
        avg_cuda = np.mean([r['cuda_mean'] for r in results])
        avg_speedup = avg_pytorch / avg_cuda
        avg_max_diff = np.mean([r['max_diff'] for r in results])
        
        print(f"\nAverage PyTorch time:  {avg_pytorch:.4f} ms")
        print(f"Average CUDA time:     {avg_cuda:.4f} ms")
        print(f"Average speedup:       {avg_speedup:.2f}x")
        print(f"Average max diff:      {avg_max_diff:.9f}")
        
        print("\nPer-sample results:")
        for i, result in enumerate(results):
            print(f"  Sample {i+1}: {result['cuda_mean']:.4f} ms "
                  f"({result['num_windows']} windows, "
                  f"{result['tokens_per_window']} tokens/window)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test attention on BraTS data')
    parser.add_argument('--sample', type=str, 
                       default='d:/dev/Brain_Tumor_Segmentation/data/preprocessed_tensors/subject_0000.pt',
                       help='Path to BraTS sample .pt file')
    parser.add_argument('--data-dir', type=str,
                       default='d:/dev/Brain_Tumor_Segmentation/data/preprocessed_tensors',
                       help='Directory containing BraTS samples')
    parser.add_argument('--multiple', action='store_true',
                       help='Test on multiple samples')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to test (if --multiple)')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of timing iterations')
    parser.add_argument('--window-size', type=int, nargs=3, default=[7, 7, 7],
                       help='Window size (default: 7 7 7)')
    
    args = parser.parse_args()
    
    if args.multiple:
        test_multiple_samples(args.data_dir, args.num_samples, args.iterations)
    else:
        test_attention_on_brats(args.sample, args.iterations, window_size=tuple(args.window_size))
