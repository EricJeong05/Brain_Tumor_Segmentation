import sys
import torch
import numpy as np
from torch.utils.cpp_extension import load
import os
import time
"""
Tests the following lines in swin_unetr.py:
520. q = q * self.scale
521. attn = q @ k.transpose(-2, -1)
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
    print("ATTENTION QK^T CALCULATION TEST")
    print("=" * 80)
    print(f"\nTest Configuration:")
    print(f"  Batch size: {batch}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Tokens per window: {tokens_per_window}")
    print(f"  Head dimension: {head_dim}")
    print(f"  Scale factor: {head_dim**-0.5:.6f}")
    
    # Create query and key tensors
    # Shape after qkv split and permute: [batch, num_heads, tokens, head_dim]
    q = torch.randn(batch, num_heads, tokens_per_window, head_dim)
    k = torch.randn(batch, num_heads, tokens_per_window, head_dim)
    
    # Apply scaling (this is done in line 520)
    scale = head_dim ** -0.5
    q_scaled = q * scale
    
    print(f"\n" + "-" * 80)
    print("INPUT TENSORS (before scaling)")
    print("-" * 80)
    print(f"\nQuery (q) shape: {q.shape}")
    print(f"Key (k) shape: {k.shape}")
    print(f"\nFirst 3 values of q[0, 0, 0, :]:")
    print(q[0, 0, 0, :3].numpy())
    print(f"\nFirst 3 values of k[0, 0, 0, :]:")
    print(k[0, 0, 0, :3].numpy())
    
    print(f"\n" + "-" * 80)
    print("SCALED QUERY (q * scale)")
    print("-" * 80)
    print(f"\nScaled Query (q_scaled) shape: {q_scaled.shape}")
    print(f"First 3 values of q_scaled[0, 0, 0, :]:")
    print(q_scaled[0, 0, 0, :3].numpy())
    
    # Compute attention scores (line 521: attn = q @ k.transpose(-2, -1))
    start_time = time.time()
    k_transposed = k.transpose(-2, -1)
    attn = q_scaled @ k_transposed
    end_time = time.time() - start_time

    print(f"Time taken for attention score computation: {end_time} seconds")

    print(f"\n" + "-" * 80)
    print("ATTENTION SCORES (q_scaled @ k^T)")
    print("-" * 80)
    print(f"\nKey transposed shape: {k_transposed.shape}")
    print(f"Attention scores shape: {attn.shape}")
    
    print(f"\n" + "=" * 80)
    print("EXAMPLE OUTPUTS")
    print("=" * 80)
    
    # Print attention matrix for first batch, first head
    print(f"\nAttention matrix [0, 0, :, :] (batch=0, head=0):")
    print(attn[0, 0].numpy())
    
    print(f"\nAttention matrix [0, 1, :, :] (batch=0, head=1):")
    print(attn[0, 1].numpy())
    
    print(f"\nAttention matrix [1, 0, :, :] (batch=1, head=0):")
    print(attn[1, 0].numpy())
    
    # Save tensors for CUDA kernel validation
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "test_data/")
    
    print(f"\n" + "-" * 80)
    print("SAVING TENSORS FOR CUDA VALIDATION")
    print("-" * 80)
    
    # Create test_data directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save as numpy arrays
    np.save(f"{save_path}q_input.npy", q.numpy())
    np.save(f"{save_path}k_input.npy", k.numpy())
    np.save(f"{save_path}q_scaled.npy", q_scaled.numpy())
    np.save(f"{save_path}attn_output.npy", attn.numpy())
    
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
    print(f"  - q_scaled.npy: shape {q_scaled.shape}")
    print(f"  - attn_output.npy: shape {attn.shape}")
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
    q_scaled = np.load(f"{path}q_scaled.npy")
    attn_output = np.load(f"{path}attn_output.npy")
    
    print(f"\nLoaded tensors:")
    print(f"  q_input shape: {q_input.shape}")
    print(f"  k_input shape: {k_input.shape}")
    print(f"  q_scaled shape: {q_scaled.shape}")
    print(f"  attn_output shape: {attn_output.shape}")
    
    return {
        'metadata': metadata,
        'q_input': q_input,
        'k_input': k_input,
        'q_scaled': q_scaled,
        'attn_output': attn_output
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
        print(f"\n✓ CUDA kernel output matches PyTorch within tolerance!")
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
        print("✓ CUDA kernel compiled successfully!")
        
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
    
    metadata = data['metadata']
    batch = metadata['batch']
    num_heads = metadata['num_heads']
    tokens = metadata['tokens_per_window']
    head_dim = metadata['head_dim']
    
    print(f"  Input query (GPU): {q_scaled_gpu.shape}")
    print(f"  Input key (GPU): {k_input_gpu.shape}")
    
    # Launch the CUDA kernel
    print("\n" + "-" * 80)
    print("Launching CUDA kernel...")

    start_time = time.time()
    try:
        # Call the CUDA kernel through the wrapper
        output_gpu = naive_attention_module.naive_attention_forward(
            q_scaled_gpu,
            k_input_gpu,
            batch,
            num_heads,
            tokens,
            head_dim
        )
        
        # Synchronize to ensure kernel execution is complete
        torch.cuda.synchronize()
        print("✓ CUDA kernel executed successfully!")
        
    except Exception as e:
        print(f"❌ Failed to execute CUDA kernel:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return False

    # Move output back to CPU and convert to numpy
    cuda_output = output_gpu.cpu().numpy()
    
    end_time = time.time() - start_time
    print(f"Time taken for CUDA kernel execution: {end_time} seconds\n")
    
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
    # Step 1: Generate test data using PyTorch attention forward operation
    print("\n[Step 1/2] Generating PyTorch test data...\n")
    
    test_pytorch_attention_forward()
    print(f"\n✓ Test completed successfully!")
    
    print("\n" + "=" * 80)
    
    # Step 2: Test CUDA kernel and verify against PyTorch output
    print("\n[Step 2/2] Testing CUDA kernel and verifying against PyTorch output...\n")
    
    success = test_cuda_kernel_attention_forward()
    
    # Final result
    if success:
        print("✅ ALL TESTS PASSED! Your CUDA kernel correctly implements the attention calculation!")
    else:
        print("❌ TEST FAILED. Check the output above for debugging information.")
