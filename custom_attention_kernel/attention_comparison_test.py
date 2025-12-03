import sys
from google_crc32c import value
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.cpp_extension import load
import os
import time
"""
Tests the following lines in swin_unetr.py:
Phase 1:
520. q = q * self.scale
521. attn = q @ k.transpose(-2, -1)

Phase 2:
533. attn = self.softmax(attn)
535. attn = self.attn_drop(attn).to(v.dtype)
536. x = (attn @ v).transpose(1, 2).reshape(b, n, c)
"""
def test_pytorch_attention_forward():
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Small test configuration
    batch = 2
    num_heads = 4
    tokens_per_window = 8  # e.g., for a 2x2x2 window
    head_dim = 16
    
    print("=" * 80)
    print("ATTENTION QK^T + SOFTMAX TEST")
    print("=" * 80)
    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Tokens per window: {tokens_per_window}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Scale factor: {head_dim**-0.5:.6f}")
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    # Create query, key, and value tensors on GPU
    # Shape after qkv split and permute: [batch, num_heads, tokens, head_dim]
    q = torch.randn(batch, num_heads, tokens_per_window, head_dim, device=device)
    k = torch.randn(batch, num_heads, tokens_per_window, head_dim, device=device)
    v = torch.randn(batch, num_heads, tokens_per_window, head_dim, device=device)
    
    # Apply scaling (this is done in line 520)
    scale = head_dim ** -0.5
    q_scaled = q * scale
    
    print(f"\n" + "-" * 80)
    print("INPUT TENSORS (before scaling)")
    print("-" * 80)
    print(f"\nQuery (q) shape: {q.shape}")
    print(f"Key (k) shape: {k.shape}")
    print(f"Value (v) shape: {v.shape}")
    print(f"\nFirst 3 values of q[0, 0, 0, :]:")
    print(q[0, 0, 0, :3].cpu().numpy())
    print(f"\nFirst 3 values of k[0, 0, 0, :]:")
    print(k[0, 0, 0, :3].cpu().numpy())
    
    print(f"\n" + "-" * 80)
    print("SCALED QUERY (q * scale)")
    print("-" * 80)
    print(f"\nScaled Query (q_scaled) shape: {q_scaled.shape}")
    print(f"First 3 values of q_scaled[0, 0, 0, :]:")
    print(q_scaled[0, 0, 0, :3].cpu().numpy())
    
    # Compute attention scores (line 521: attn = q @ k.transpose(-2, -1))
    k_transposed = k.transpose(-2, -1)
    attn_scores = q_scaled @ k_transposed

    print(f"\n" + "-" * 80)
    print("RAW ATTENTION SCORES (q_scaled @ k^T) - BEFORE SOFTMAX")
    print("-" * 80)
    print(f"\nKey transposed shape: {k_transposed.shape}")
    print(f"Attention scores shape: {attn_scores.shape}")
    print(f"\nFirst row of attention scores [0, 0, 0, :]:")
    print(attn_scores[0, 0, 0, :].cpu().numpy())
    
    # Compute softmax (line 533: attn = self.softmax(attn))
    softmax = nn.Softmax(dim=-1)
    attn = softmax(attn_scores)

    print(f"\n" + "-" * 80)
    print("ATTENTION PROBABILITIES (after softmax)")
    print("-" * 80)
    print(f"\nAttention probabilities shape: {attn.shape}")
    print(f"\nFirst row of attention probabilities [0, 0, 0, :] (should sum to ~1.0):")
    first_row = attn[0, 0, 0, :].cpu().numpy()
    print(first_row)
    print(f"Row sum: {np.sum(first_row):.6f}")
    
    # Compute attention output (line 536: x = (attn @ v).transpose(1, 2).reshape(b, n, c))
    attn_output = attn @ v  # [batch, num_heads, tokens, head_dim]
    
    print(f"\n" + "-" * 80)
    print("ATTENTION OUTPUT (attn @ v)")
    print("-" * 80)
    print(f"\nAttention output shape: {attn_output.shape}")
    print(f"First token output [0, 0, 0, :3] (first head, first token, first 3 dims):")
    print(attn_output[0, 0, 0, :3].cpu().numpy())

    # Synchronize if on GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    print(f"\n" + "=" * 80)
    print("EXAMPLE OUTPUTS")
    print("=" * 80)
    
    # Print attention probability matrices (after softmax - each row sums to 1)
    print(f"\nAttention probability matrix [0, 0, :, :] (batch=0, head=0):")
    attn_00 = attn[0, 0].cpu().numpy()
    print(attn_00)
    print(f"Row sums: {np.sum(attn_00, axis=1)}")
    
    print(f"\nAttention probability matrix [0, 1, :, :] (batch=0, head=1):")
    attn_01 = attn[0, 1].cpu().numpy()
    print(attn_01)
    
    print(f"\nAttention probability matrix [1, 0, :, :] (batch=1, head=0):")
    attn_10 = attn[1, 0].cpu().numpy()
    print(attn_10)
    
    # Save tensors for CUDA kernel validation
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "test_data/")
    
    print(f"\n" + "-" * 80)
    print("SAVING TENSORS FOR CUDA VALIDATION")
    print("-" * 80)
    
    # Create test_data directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save as numpy arrays (move to CPU first)
    np.save(f"{save_path}q_input.npy", q.cpu().numpy())
    np.save(f"{save_path}k_input.npy", k.cpu().numpy())
    np.save(f"{save_path}v_input.npy", v.cpu().numpy())
    np.save(f"{save_path}q_scaled.npy", q_scaled.cpu().numpy())
    np.save(f"{save_path}attn_probs.npy", attn.cpu().numpy())
    np.save(f"{save_path}attn_output.npy", attn_output.cpu().numpy())
    
    # Save metadata
    metadata = {
        'batch': batch,
        'num_heads': num_heads,
        'tokens_per_window': tokens_per_window,
        'head_dim': head_dim,
        'scale': scale
    }
    np.save(f"{save_path}metadata.npy", metadata)
    
    print(f"Saved tensors to: {save_path}")
    print(f"  - q_input.npy: shape {q.shape}")
    print(f"  - k_input.npy: shape {k.shape}")
    print(f"  - v_input.npy: shape {v.shape}")
    print(f"  - q_scaled.npy: shape {q_scaled.shape}")
    print(f"  - attn_probs.npy: shape {attn.shape}")
    print(f"  - attn_output.npy: shape {attn_output.shape}")
    print(f"  - metadata.npy: configuration parameters")

    return True
    
"""
Load all saved test tensors and metadata.
"""
def load_test_data(path="d:/dev/Brain_Tumor_Segmentation/models/custom_swin_unetr/cuda/test_data/"):
    print("=" * 80)
    print("LOADING TEST DATA FOR CUDA KERNEL VALIDATION")
    print("=" * 80)
    
    # Load metadata
    metadata = np.load(f"{path}metadata.npy", allow_pickle=True).item()
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {metadata['batch']}")
    print(f"  Number of heads: {metadata['num_heads']}")
    print(f"  Tokens per window: {metadata['tokens_per_window']}")
    print(f"  Head dimension: {metadata['head_dim']}")
    print(f"  Scale factor: {metadata['scale']:.6f}")
    
    # Load tensors
    q_input = np.load(f"{path}q_input.npy")
    k_input = np.load(f"{path}k_input.npy")
    v_input = np.load(f"{path}v_input.npy")
    q_scaled = np.load(f"{path}q_scaled.npy")
    attn_probs = np.load(f"{path}attn_probs.npy")
    attn_output = np.load(f"{path}attn_output.npy")
    
    print(f"\nLoaded tensors:")
    print(f"  q_input shape: {q_input.shape}")
    print(f"  k_input shape: {k_input.shape}")
    print(f"  v_input shape: {v_input.shape}")
    print(f"  q_scaled shape: {q_scaled.shape}")
    print(f"  attn_probs shape: {attn_probs.shape}")
    print(f"  attn_output shape: {attn_output.shape}")
    
    return {
        'metadata': metadata,
        'q_input': q_input,
        'k_input': k_input,
        'v_input': v_input,
        'q_scaled': q_scaled,
        'attn_probs': attn_probs,
        'attn_output': attn_output
    }


"""
Benchmark PyTorch and CUDA attention implementations.

Args:
    batch: Batch size
    num_heads: Number of attention heads
    tokens_per_window: Number of tokens per window
    head_dim: Dimension of each head
    num_iterations: Number of timing iterations
    num_warmup: Number of warmup iterations (not included in timing)
"""
def benchmark_attention(batch=2, num_heads=4, tokens_per_window=8, head_dim=16, 
                       num_iterations=10000, num_warmup=10):
    print("\n" + "=" * 80)
    print("ATTENTION PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Tokens per window: {tokens_per_window}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Warmup iterations: {num_warmup}")
    print(f"  Timing iterations: {num_iterations}")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, cannot run benchmark")
        return None
    
    device = torch.device('cuda')
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create test tensors on GPU
    scale = head_dim ** -0.5
    q = torch.randn(batch, num_heads, tokens_per_window, head_dim, device=device)
    k = torch.randn(batch, num_heads, tokens_per_window, head_dim, device=device)
    v = torch.randn(batch, num_heads, tokens_per_window, head_dim, device=device)
    q_scaled = q * scale
    
    # ========== PyTorch Benchmark ==========
    print("\n" + "-" * 80)
    print("PyTorch Attention (GPU) - QK^T + Softmax")
    print("-" * 80)
    
    softmax = nn.Softmax(dim=-1)
    
    # Warmup
    for _ in range(num_warmup):
        attn_scores = q_scaled @ k.transpose(-2, -1)
        attn_pytorch = softmax(attn_scores)
        _ = attn_pytorch @ v
        torch.cuda.synchronize()
    
    # Timed runs
    pytorch_times = []
    for _ in range(num_iterations):
        torch.cuda.synchronize()
        start = time.time()
        attn_scores = q_scaled @ k.transpose(-2, -1)
        attn_pytorch = softmax(attn_scores)
        output_pytorch = attn_pytorch @ v
        torch.cuda.synchronize()
        end = time.time()
        pytorch_times.append(end - start)
    
    pytorch_mean = np.mean(pytorch_times) * 1000  # Convert to ms
    pytorch_std = np.std(pytorch_times) * 1000
    pytorch_min = np.min(pytorch_times) * 1000
    pytorch_max = np.max(pytorch_times) * 1000
    
    print(f"  Mean time: {pytorch_mean:.4f} ms")
    print(f"  Std dev:   {pytorch_std:.4f} ms")
    print(f"  Min time:  {pytorch_min:.4f} ms")
    print(f"  Max time:  {pytorch_max:.4f} ms")
    
    # ========== Custom CUDA Kernel Benchmark ==========
    print("\n" + "-" * 80)
    print("Custom CUDA Attention Kernel (GPU) - QK^T + Softmax")
    print("-" * 80)
    
    # Compile CUDA kernel
    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        print("  Compiling CUDA kernel...")
        naive_attention_module = load(
            name="naive_attention",
            sources=[
                os.path.join(cuda_dir, "kernels/naive_attention_forward.cpp"),
                os.path.join(cuda_dir, "kernels/naive_attention_forward.cu")
            ],
            verbose=False
        )
        print("  ✅ Compilation successful")
    except Exception as e:
        print(f"  ❌ Failed to compile: {e}")
        return
    
    # Data is already on GPU
    
    # Warmup
    for _ in range(num_warmup):
        _ = naive_attention_module.naive_attention_forward(
            q_scaled, k, v, batch, num_heads, tokens_per_window, head_dim
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
        #     q_scaled, k, v, batch, num_heads, tokens_per_window, head_dim
        # )
        torch.cuda.synchronize()
        end = time.time()
        cuda_times.append(end - start)
    
    cuda_mean = np.mean(cuda_times) * 1000  # Convert to ms
    cuda_std = np.std(cuda_times) * 1000
    cuda_min = np.min(cuda_times) * 1000
    cuda_max = np.max(cuda_times) * 1000
    
    print(f"  Mean time: {cuda_mean:.4f} ms")
    print(f"  Std dev:   {cuda_std:.4f} ms")
    print(f"  Min time:  {cuda_min:.4f} ms")
    print(f"  Max time:  {cuda_max:.4f} ms")
    
    # ========== Results Summary ==========
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    speedup = pytorch_mean / cuda_mean
    
    print(f"\nPyTorch (GPU):        {pytorch_mean:.4f} ms ± {pytorch_std:.4f} ms")
    print(f"Custom CUDA Kernel:   {cuda_mean:.4f} ms ± {cuda_std:.4f} ms")
    
    if speedup > 1:
        print(f"✅ Custom kernel is {speedup:.2f}x faster than PyTorch GPU!")
    else:
        speedup = -1/speedup
        print(f"⚠️  PyTorch GPU is {-speedup:.2f}x faster than custom kernel")
    
    # Verify correctness
    print("\n" + "-" * 80)
    print("Verifying correctness...")
    output_pytorch_cpu = output_pytorch.cpu().numpy()
    output_cuda_cpu = output_cuda.cpu().numpy()
    
    max_diff = np.max(np.abs(output_pytorch_cpu - output_cuda_cpu))
    print(f"  Max difference: {max_diff:.9f}")
    
    if max_diff < 1e-5:
        print("  ✅ Outputs match within tolerance!")
    else:
        print("  ⚠️  Outputs differ - check implementation!")
    
    print("=" * 80 + "\n")
    
    return {
        'pytorch_mean': pytorch_mean,
        'pytorch_std': pytorch_std,
        'cuda_mean': cuda_mean,
        'cuda_std': cuda_std,
        'speedup': speedup,
        'max_diff': max_diff
    }


"""
Compare CUDA kernel output with expected PyTorch output.

Args:
    cuda_output: Output from your CUDA kernel (numpy array)
    expected_output: Expected output from PyTorch (numpy array)
    tolerance: Maximum allowed difference
"""
def verify_cuda_output(cuda_output, expected_output, tolerance=1e-5):
    print("\n" + "=" * 80)
    print("CUDA KERNEL VALIDATION")
    print("=" * 80)
    
    # Ensure same shape
    if cuda_output.shape != expected_output.shape:
        print(f"\n❌ SHAPE MISMATCH!")
        print(f"  CUDA output shape: {cuda_output.shape}")
        print(f"  Expected shape: {expected_output.shape}")
        return False
    
    # Compute differences
    diff = np.abs(cuda_output - expected_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\nDifference statistics:")
    print(f"  Max difference:  {max_diff:.9f}")
    print(f"  Mean difference: {mean_diff:.9f}")
    print(f"  Tolerance:       {tolerance:.9f}")
    
    # Check if within tolerance
    if max_diff <= tolerance:
        print(f"\n✅ CUDA kernel output matches PyTorch within tolerance!")
        print(f"  All values are within {tolerance} of expected values.")
        return True
    else:
        print(f"\n❌ CUDA kernel output DOES NOT match PyTorch!")
        print(f"  Max difference {max_diff:.9f} exceeds tolerance {tolerance:.9f}")
        
        # Find worst mismatches
        worst_indices = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"\n  Worst mismatch at indices {worst_indices}:")
        print(f"    CUDA output:  {cuda_output[worst_indices]:.6f}")
        print(f"    Expected:     {expected_output[worst_indices]:.6f}")
        print(f"    Difference:   {diff[worst_indices]:.9f}")
        
        return False


"""
Print detailed information about a tensor.
"""
def print_tensor_details(tensor, name, max_elements=10):    
    print(f"\n{name}:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Min: {np.min(tensor):.6f}, Max: {np.max(tensor):.6f}")
    print(f"  Mean: {np.mean(tensor):.6f}, Std: {np.std(tensor):.6f}")
    
    if tensor.size <= max_elements:
        print(f"  All values:\n{tensor}")
    else:
        flat = tensor.flatten()
        print(f"  First {max_elements} values: {flat[:max_elements]}")


"""
Compile and test the naive_attention CUDA kernel against PyTorch output.
"""
def test_cuda_kernel_attention_forward():    
    print("=" * 80)
    print("TESTING CUDA KERNEL")
    print("=" * 80)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available on this system!")
        print("   Make sure you have a CUDA-capable GPU and PyTorch with CUDA support.")
        return False
    
    # Get the directory of this script
    cuda_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\nCUDA kernel directory: {cuda_dir}")
    print("Compiling CUDA kernel... (this may take a minute)")
    
    try:
        # Compile the CUDA kernel using PyTorch's JIT compiler
        naive_attention_module = load(
            name="naive_attention",
            sources=[
                os.path.join(cuda_dir, "kernels/naive_attention_forward.cpp"),
                os.path.join(cuda_dir, "kernels/naive_attention_forward.cu")
            ],
            verbose=True
        )
        print("✅ CUDA kernel compiled successfully!")
        
    except Exception as e:
        print(f"❌ Failed to compile CUDA kernel:")
        print(f"   {e}")
        return False
    
    # Load test data
    data = load_test_data(path=os.path.join(cuda_dir, "test_data/"))
    
    # Convert numpy arrays to PyTorch tensors and move to GPU
    print("\n" + "-" * 80)
    print("Preparing data for CUDA kernel...")
    
    q_scaled_gpu = torch.from_numpy(data['q_scaled']).cuda()
    k_input_gpu = torch.from_numpy(data['k_input']).cuda()
    v_input_gpu = torch.from_numpy(data['v_input']).cuda()
    
    metadata = data['metadata']
    batch = metadata['batch']
    num_heads = metadata['num_heads']
    tokens = metadata['tokens_per_window']
    head_dim = metadata['head_dim']
    
    print(f"  Input query (GPU): {q_scaled_gpu.shape}")
    print(f"  Input key (GPU): {k_input_gpu.shape}")
    print(f"  Input value (GPU): {v_input_gpu.shape}")
    
    # Launch the CUDA kernel
    print("\n" + "-" * 80)
    print("Launching CUDA kernel...")

    try:
        # Call the CUDA kernel through the wrapper
        output_gpu = naive_attention_module.naive_attention_forward(
            q_scaled_gpu,
            k_input_gpu,
            v_input_gpu,
            batch,
            num_heads,
            tokens,
            head_dim
        )
        
        # Synchronize to ensure kernel execution is complete
        torch.cuda.synchronize()
        print("✅ CUDA kernel executed successfully!")
        
    except Exception as e:
        print(f"❌ Failed to execute CUDA kernel:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return False

    # Move output back to CPU and convert to numpy
    cuda_output = output_gpu.cpu().numpy()
    
    print(f"  Output shape: {cuda_output.shape}")
    
    # Compare with PyTorch output
    success = verify_cuda_output(cuda_output, data['attn_output'], tolerance=1e-5)
    
    if success:
        print("\n" + "=" * 80)
        print("🎉 SUCCESS! CUDA kernel matches PyTorch implementation!")
        print("=" * 80  + "\n")
    else:
        print("\n" + "=" * 80)
        print("❌ FAILED! CUDA kernel output differs from PyTorch")
        print("=" * 80 + "\n")
        
        # Print a few sample values for debugging
        print("\nSample values for debugging:")
        print(f"  CUDA output[0,0,0,:3]:    {cuda_output[0,0,0,:3]}")
        print(f"  Expected output[0,0,0,:3]: {data['attn_output'][0,0,0,:3]}")
    
    return success

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test and benchmark attention implementations')
    parser.add_argument('--benchmark', action='store_true', 
                       help='Run performance benchmark')
    parser.add_argument('--benchmark-sizes', action='store_true',
                       help='Run benchmark across multiple problem sizes')
    parser.add_argument('--iterations', type=int, default=10000,
                       help='Number of timing iterations (default: 10000)')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations (default: 10)')
    
    args = parser.parse_args()
    
    if args.benchmark_sizes:
        # Benchmark multiple problem sizes
        print("\n" + "=" * 80)
        print("MULTI-SIZE BENCHMARK")
        print("=" * 80)
        
        test_configs = [
            (1, 4, 8, 16),      # Small: Original test size
            (2, 6, 49, 32),     # Medium: 7x7 window
            (4, 12, 343, 32),   # Large: 7x7x7 window
            (8, 24, 343, 32),   # Very Large: Higher batch/heads
        ]
        
        results = []
        for batch, num_heads, tokens, head_dim in test_configs:
            print(f"\n{'='*80}")
            print(f"Testing: batch={batch}, heads={num_heads}, tokens={tokens}, dim={head_dim}")
            print(f"{'='*80}")
            
            result = benchmark_attention(
                batch=batch,
                num_heads=num_heads,
                tokens_per_window=tokens,
                head_dim=head_dim,
                num_iterations=args.iterations,
                num_warmup=args.warmup
            )
            
            if result:
                results.append({
                    'config': (batch, num_heads, tokens, head_dim),
                    'speedup': result['speedup'],
                    'pytorch_ms': result['pytorch_mean'],
                    'cuda_ms': result['cuda_mean']
                })
        
        # Summary table
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"\n{'Config':<30} {'PyTorch (ms)':<15} {'CUDA (ms)':<15} {'Speedup':<10}")
        print("-" * 80)
        
        for r in results:
            config_str = f"B{r['config'][0]}_H{r['config'][1]}_T{r['config'][2]}_D{r['config'][3]}"
            print(f"{config_str:<30} {r['pytorch_ms']:<15.4f} {r['cuda_ms']:<15.4f} {r['speedup']:<10.2f}x")
        
        print("=" * 80 + "\n")
        
    elif args.benchmark:
        # Single benchmark run
        benchmark_attention(
            num_iterations=args.iterations,
            num_warmup=args.warmup
        )
        
    else:
        # Standard correctness test
        # Step 1: Generate test data using PyTorch attention forward operation
        print("\n[Step 1/2] Generating PyTorch test data...\n")
        
        test_pytorch_attention_forward()
        print(f"\n✅ Test completed successfully!")
        
        print("\n" + "=" * 80)
        
        # Step 2: Test CUDA kernel and verify against PyTorch output
        print("\n[Step 2/2] Testing CUDA kernel and verifying against PyTorch output...\n")
        
        success = test_cuda_kernel_attention_forward()
        
        # Final result
        if success:
            print("✅ ALL TESTS PASSED! Your CUDA kernel correctly implements the attention calculation!")
            print("\n💡 Tip: Run with --benchmark to see performance comparison")
            print("        Run with --benchmark-sizes to test multiple problem sizes")
        else:
            print("❌ TEST FAILED. Check the output above for debugging information.")
