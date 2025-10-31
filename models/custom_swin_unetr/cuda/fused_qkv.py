import os
import ctypes
import torch
import numpy as np

# Load the CUDA DLL
current_dir = os.path.dirname(os.path.abspath(__file__))
cuda_dll = ctypes.CDLL(os.path.join(current_dir, "fused_qkv_kernel.dll"))

# Define function argument types
cuda_dll.launch_fused_qkv.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # x
    ctypes.POINTER(ctypes.c_float),  # weight
    ctypes.POINTER(ctypes.c_float),  # bias
    ctypes.POINTER(ctypes.c_float),  # output
    ctypes.c_int,                    # batch_size
    ctypes.c_int,                    # seq_length
    ctypes.c_int,                    # dim
    ctypes.c_int,                    # num_heads
]

def fused_qkv_projection(x, weight, bias, num_heads):
    """
    Fused QKV projection using CUDA.
    
    Args:
        x: Input tensor of shape (batch_size, seq_length, dim)
        weight: Weight matrix of shape (3 * dim, dim)
        bias: Bias vector of shape (3 * dim,)
        num_heads: Number of attention heads
    
    Returns:
        Output tensor of shape (3, batch_size, num_heads, seq_length, head_dim)
    """
    batch_size, seq_length, dim = x.shape
    head_dim = dim // num_heads
    
    # Create output tensor
    output = torch.empty((3, batch_size, num_heads, seq_length, head_dim),
                        device='cuda', dtype=torch.float32)
    
    # Get pointers to the GPU memory
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_float))
    weight_ptr = ctypes.cast(weight.data_ptr(), ctypes.POINTER(ctypes.c_float))
    bias_ptr = ctypes.cast(bias.data_ptr(), ctypes.POINTER(ctypes.c_float))
    output_ptr = ctypes.cast(output.data_ptr(), ctypes.POINTER(ctypes.c_float))
    
    # Launch CUDA kernel
    cuda_dll.launch_fused_qkv(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        batch_size,
        seq_length,
        dim,
        num_heads
    )
    
    return output