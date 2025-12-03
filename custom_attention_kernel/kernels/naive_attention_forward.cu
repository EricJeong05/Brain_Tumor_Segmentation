// Fused attention kernel: computes QK^T and applies softmax
// Uses tiling to reduce global memory accesses and improve performance
#define TILE_SIZE 32  // Tile size for shared memory (optimized for NVIDIA RTX 4060Ti)
#define WARP_SIZE 32

__global__ void naive_attention_forward(
    const float* __restrict__ query,  // [batch, num_heads, tokens, head_dim] - already scaled!
    const float* __restrict__ key,    // [batch, num_heads, tokens, head_dim]
    float* __restrict__ output,       // [batch, num_heads, tokens, tokens]
    int batch, 
    int num_heads,
    int tokens_per_window,
    int head_dim)
{
    // Shared memory tiles for query and key
    __shared__ float tile_query[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_key[TILE_SIZE][TILE_SIZE];
    
    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block indices give us position in output matrix
    int block_row = blockIdx.y;  // which row of tiles
    int block_col = blockIdx.x;  // which column of tiles
    
    // Batch and head indices
    int bh = blockIdx.z;  // combined batch * num_heads index
    int b = bh / num_heads;
    int h = bh % num_heads;
    
    // Global row and column in output matrix
    int row = block_row * TILE_SIZE + ty;
    int col = block_col * TILE_SIZE + tx;
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Base offset for this batch and head
    int bh_offset = (b * num_heads + h) * tokens_per_window * head_dim;
    
    // Number of tiles needed to cover head_dim
    int num_tiles = (head_dim + TILE_SIZE - 1) / TILE_SIZE;
    
    // ===== Step 1: Compute QK^T =====
    // Loop over tiles along the head_dim dimension
    for (int tile = 0; tile < num_tiles; tile++) {
        // Calculate dimension index for this tile
        int dim_idx = tile * TILE_SIZE + tx;
        
        // Load query tile into shared memory
        // query[b][h][row][dim_idx]
        if (row < tokens_per_window && dim_idx < head_dim) {
            int query_idx = bh_offset + row * head_dim + dim_idx;
            tile_query[ty][tx] = query[query_idx];
        } else {
            tile_query[ty][tx] = 0.0f;
        }
        
        // Load key tile into shared memory (note: we load key[col][dim_idx] for transpose)
        // key[b][h][col][dim_idx]
        dim_idx = tile * TILE_SIZE + ty;  // use ty for dimension in key
        if (col < tokens_per_window && dim_idx < head_dim) {
            int key_idx = bh_offset + col * head_dim + dim_idx;
            tile_key[ty][tx] = key[key_idx];
        } else {
            tile_key[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure tile is loaded
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_query[ty][k] * tile_key[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // ===== Step 2: Write QK^T to global memory (needed for proper softmax across all blocks) =====
    if (row < tokens_per_window && col < tokens_per_window) {
        int output_idx = bh_offset / head_dim * tokens_per_window + row * tokens_per_window + col;
        output[output_idx] = sum;
    }
}

// Warp-level reduction for max using shuffle operations
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level reduction for sum using shuffle operations
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized softmax kernel using warp-level primitives
__global__ void softmax_attention_kernel(
    float* __restrict__ attn,  // [batch, num_heads, tokens, tokens]
    int batch,
    int num_heads,
    int tokens_per_window)
{
    // Each block processes one row of the attention matrix
    int row = blockIdx.x;
    int bh = blockIdx.y;  // batch * num_heads index
    
    if (row >= tokens_per_window) return;
    
    int b = bh / num_heads;
    int h = bh % num_heads;
    
    // Calculate row offset in attention matrix
    int row_offset = (b * num_heads + h) * tokens_per_window * tokens_per_window + row * tokens_per_window;
    
    // Thread and warp identification
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    
    __shared__ float warp_maxs[8];  // Max from each warp (support up to 8 warps = 256 threads)
    __shared__ float warp_sums[8];  // Sum from each warp
    __shared__ float shared_max;
    __shared__ float shared_sum;
    
    // ===== Phase 1: Find max value in row (for numerical stability) =====
    float thread_max = -INFINITY;
    for (int i = tid; i < tokens_per_window; i += blockDim.x) {
        thread_max = fmaxf(thread_max, attn[row_offset + i]);
    }
    
    // Warp-level reduction for max
    float warp_max = warp_reduce_max(thread_max);
    
    // First thread of each warp writes to shared memory
    if (lane_id == 0) {
        warp_maxs[warp_id] = warp_max;
    }
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? warp_maxs[lane_id] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            shared_max = val;
        }
    }
    __syncthreads();
    
    // ===== Phase 2: Compute exp(x - max) and sum =====
    float thread_sum = 0.0f;
    for (int i = tid; i < tokens_per_window; i += blockDim.x) {
        float exp_val = expf(attn[row_offset + i] - shared_max);
        attn[row_offset + i] = exp_val;  // Store exp values back
        thread_sum += exp_val;
    }
    
    // Warp-level reduction for sum
    float warp_sum = warp_reduce_sum(thread_sum);
    
    // First thread of each warp writes to shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction across warps (done by first warp)
    if (warp_id == 0) {
        float val = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            shared_sum = val;
        }
    }
    __syncthreads();
    
    // ===== Phase 3: Normalize by sum =====
    for (int i = tid; i < tokens_per_window; i += blockDim.x) {
        attn[row_offset + i] = attn[row_offset + i] / shared_sum;
    }
}

// Kernel for attention-value matrix multiplication: output = attn @ v
// Uses tiling to optimize memory access patterns
__global__ void attention_value_matmul(
    const float* __restrict__ attn,   // [batch, num_heads, tokens, tokens] - attention probabilities
    const float* __restrict__ value,  // [batch, num_heads, tokens, head_dim]
    float* __restrict__ output,       // [batch, num_heads, tokens, head_dim]
    int batch,
    int num_heads,
    int tokens_per_window,
    int head_dim)
{
    // Shared memory tiles for attention and value
    __shared__ float tile_attn[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_value[TILE_SIZE][TILE_SIZE];
    
    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block indices give us position in output matrix
    int block_row = blockIdx.y;  // which row of tiles (tokens dimension)
    int block_col = blockIdx.x;  // which column of tiles (head_dim dimension)
    
    // Batch and head indices
    int bh = blockIdx.z;  // combined batch * num_heads index
    int b = bh / num_heads;
    int h = bh % num_heads;
    
    // Global row and column in output matrix
    int row = block_row * TILE_SIZE + ty;  // token index
    int col = block_col * TILE_SIZE + tx;  // head_dim index
    
    // Accumulator for dot product
    float sum = 0.0f;
    
    // Base offsets for this batch and head
    int attn_bh_offset = (b * num_heads + h) * tokens_per_window * tokens_per_window;
    int value_bh_offset = (b * num_heads + h) * tokens_per_window * head_dim;
    
    // Number of tiles needed to cover tokens_per_window dimension
    int num_tiles = (tokens_per_window + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over tiles along the tokens dimension
    for (int tile = 0; tile < num_tiles; tile++) {
        // Calculate token index for this tile
        int token_idx = tile * TILE_SIZE + tx;
        
        // Load attention tile into shared memory
        // attn[b][h][row][token_idx]
        if (row < tokens_per_window && token_idx < tokens_per_window) {
            int attn_idx = attn_bh_offset + row * tokens_per_window + token_idx;
            tile_attn[ty][tx] = attn[attn_idx];
        } else {
            tile_attn[ty][tx] = 0.0f;
        }
        
        // Load value tile into shared memory
        // value[b][h][token_idx][col]
        token_idx = tile * TILE_SIZE + ty;  // use ty for token in value
        if (token_idx < tokens_per_window && col < head_dim) {
            int value_idx = value_bh_offset + token_idx * head_dim + col;
            tile_value[ty][tx] = value[value_idx];
        } else {
            tile_value[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure tile is loaded
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tile_attn[ty][k] * tile_value[k][tx];
        }
        
        // Synchronize before loading next tile
        __syncthreads();
    }
    
    // Write result to global memory
    if (row < tokens_per_window && col < head_dim) {
        int output_idx = value_bh_offset + row * head_dim + col;
        output[output_idx] = sum;
    }
}

// Kernel launcher function (called from C++ wrapper)
void launch_naive_attention_forward(
    const float* query,
    const float* key,
    const float* value,
    float* attn_probs,
    float* output,
    int batch,
    int num_heads,
    int tokens_per_window,
    int head_dim,
    int blocks,
    int threads_per_block)
{
    // ===== Step 1: Compute QK^T using tiled matrix multiplication =====
    // Configure grid and block dimensions for tiled kernel
    // Each block handles a TILE_SIZE x TILE_SIZE output tile
    dim3 blockDim_qk(TILE_SIZE, TILE_SIZE);
    
    // Grid dimensions: 
    // x-axis: number of column tiles (tokens_per_window / TILE_SIZE)
    // y-axis: number of row tiles (tokens_per_window / TILE_SIZE)
    // z-axis: batch * num_heads
    int grid_x = (tokens_per_window + TILE_SIZE - 1) / TILE_SIZE;
    int grid_y = (tokens_per_window + TILE_SIZE - 1) / TILE_SIZE;
    int grid_z = batch * num_heads;
    
    dim3 gridDim_qk(grid_x, grid_y, grid_z);
    
    naive_attention_forward<<<gridDim_qk, blockDim_qk>>>(
        query,
        key,
        attn_probs,
        batch,
        num_heads,
        tokens_per_window,
        head_dim
    );
    
    // ===== Step 2: Apply softmax row-wise =====
    // Each block processes one row, threads cooperate within the row
    // Grid: (tokens_per_window, batch * num_heads)
    // Block: up to 256 threads per row (adjust based on tokens_per_window)
    int threads_per_row = min(256, (tokens_per_window + 31) / 32 * 32);  // Round up to warp size
    dim3 gridDim_softmax(tokens_per_window, batch * num_heads);
    dim3 blockDim_softmax(threads_per_row);
    
    softmax_attention_kernel<<<gridDim_softmax, blockDim_softmax>>>(
        attn_probs,
        batch,
        num_heads,
        tokens_per_window
    );
    
    // ===== Step 3: Compute attention-value matrix multiplication (attn @ v) =====
    // Configure grid and block dimensions for tiled matrix multiplication
    // Each block handles a TILE_SIZE x TILE_SIZE output tile
    dim3 blockDim_av(TILE_SIZE, TILE_SIZE);
    
    // Grid dimensions:
    // x-axis: number of column tiles (head_dim / TILE_SIZE)
    // y-axis: number of row tiles (tokens_per_window / TILE_SIZE)
    // z-axis: batch * num_heads
    int grid_av_x = (head_dim + TILE_SIZE - 1) / TILE_SIZE;
    int grid_av_y = (tokens_per_window + TILE_SIZE - 1) / TILE_SIZE;
    int grid_av_z = batch * num_heads;
    
    dim3 gridDim_av(grid_av_x, grid_av_y, grid_av_z);
    
    attention_value_matmul<<<gridDim_av, blockDim_av>>>(
        attn_probs,
        value,
        output,
        batch,
        num_heads,
        tokens_per_window,
        head_dim
    );
}