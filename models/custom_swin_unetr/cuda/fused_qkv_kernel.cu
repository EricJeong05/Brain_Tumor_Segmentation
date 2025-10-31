#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

// Compile with: "nvcc -o edge_detect.dll edge_detect.cu -shared" using x64 Native Tools Command Prompt for VS 2022

template <typename scalar_t>
__global__ void fused_qkv_kernel(
    const scalar_t* __restrict__ x,        // Input tensor (B = batch size, N = sequence length, C = embedding dim)
    const scalar_t* __restrict__ weight,   // Weight matrix (3*C, C)
    const scalar_t* __restrict__ bias,     // Bias vector (3*C)
    scalar_t* __restrict__ output,         // Output tensor (3, B, H = num_heads, N, D = head_dim)
    const int batch_size,
    const int seq_length,
    const int dim,
    const int num_heads
) {
    // Calculate head dimension
    const int head_dim = dim / num_heads; // Dimension per attention head
    
    // Calculate indices
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; // Global thread ID
    const int total_elements = batch_size * seq_length * num_heads * head_dim; // Total elements to process
    
    if (tid < total_elements) {
        // Decode indices
        const int head_dim_idx = tid % head_dim; // Position within head dimension
        const int tmp1 = tid / head_dim; // Temporary for further division
        const int seq_idx = tmp1 % seq_length; // Sequence position
        const int tmp2 = tmp1 / seq_length; // Temporary for batch/head calc
        const int head_idx = tmp2 % num_heads; // Head number
        const int batch_idx = tmp2 / num_heads; // Batch number
        
        // Process Q, K, V
        for (int qkv = 0; qkv < 3; qkv++) { // Loop for Query, Key, Value projections
            float sum = 0.0f;   // Accumulator for matrix multiplication
            
            // Compute matrix multiplication for this element
            for (int i = 0; i < dim; i++) {
                // Input access pattern: [batch, sequence, feature]
                const float x_val = x[batch_idx * seq_length * dim + seq_idx * dim + i];
                // Weight access pattern: [qkv, input_dim, output_dim]
                const float w_val = weight[qkv * dim * dim + i * dim + head_idx * head_dim + head_dim_idx];
                sum += x_val * w_val;
            }
            
            // Add bias
            sum += bias[qkv * dim + head_idx * head_dim + head_dim_idx];
            
            // Calculate output offset
            const int out_offset = qkv * batch_size * num_heads * seq_length * head_dim +
                                 batch_idx * num_heads * seq_length * head_dim +
                                 head_idx * seq_length * head_dim +
                                 seq_idx * head_dim +
                                 head_dim_idx;
            
            output[out_offset] = sum;
        }
    }
}

// C-style interface function
extern "C" __declspec(dllexport) void launch_fused_qkv(
    const float* x,
    const float* weight,
    const float* bias,
    float* output,
    const int batch_size,
    const int seq_length,
    const int dim,
    const int num_heads
) {
    const int threads = 256; // Threads per block
    const int total_elements = batch_size * seq_length * num_heads * (dim / num_heads);
    const int blocks = (total_elements + threads - 1) / threads; // Number of blocks
    
    fused_qkv_kernel<float><<<blocks, threads>>>(
        x, weight, bias, output,
        batch_size, seq_length, dim, num_heads
    );
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}