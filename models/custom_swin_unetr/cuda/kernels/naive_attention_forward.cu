// Tiled attention kernel: computes attn = q @ k.transpose(-2, -1) using shared memory
// Uses tiling to reduce global memory accesses and improve performance
#define TILE_SIZE 32  // Tile size for shared memory (optimized for NVIDIA RTX 4060Ti)

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
    
    // Write result to output
    if (row < tokens_per_window && col < tokens_per_window) {
        int output_idx = bh_offset / head_dim * tokens_per_window + row * tokens_per_window + col;
        output[output_idx] = sum;
    }
}

// Kernel launcher function (called from C++ wrapper)
void launch_naive_attention_forward(
    const float* query,
    const float* key,
    float* output,
    int batch,
    int num_heads,
    int tokens_per_window,
    int head_dim,
    int blocks,
    int threads_per_block)
{
    // Configure grid and block dimensions for tiled kernel
    // Each block handles a TILE_SIZE x TILE_SIZE output tile
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    
    // Grid dimensions: 
    // x-axis: number of column tiles (tokens_per_window / TILE_SIZE)
    // y-axis: number of row tiles (tokens_per_window / TILE_SIZE)
    // z-axis: batch * num_heads
    int grid_x = (tokens_per_window + TILE_SIZE - 1) / TILE_SIZE;
    int grid_y = (tokens_per_window + TILE_SIZE - 1) / TILE_SIZE;
    int grid_z = batch * num_heads;
    
    dim3 gridDim(grid_x, grid_y, grid_z);
    
    naive_attention_forward<<<gridDim, blockDim>>>(
        query,
        key,
        output,
        batch,
        num_heads,
        tokens_per_window,
        head_dim
    );
}