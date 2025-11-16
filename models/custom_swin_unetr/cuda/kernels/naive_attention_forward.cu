// Naive attention kernel: computes attn = q @ k.transpose(-2, -1)
// Each thread computes one element of the output attention matrix
__global__ void naive_attention_forward(
    const float* __restrict__ query,  // [batch, num_heads, tokens, head_dim] - already scaled!
    const float* __restrict__ key,    // [batch, num_heads, tokens, head_dim]
    float* __restrict__ output,       // [batch, num_heads, tokens, tokens]
    int batch, 
    int num_heads,
    int tokens_per_window,
    int head_dim)
{
    // Each thread computes output[b][h][i][j]
    // This is the dot product of query[b][h][i] and key[b][h][j]
    
    // Thread index tells us which output element to compute
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Total number of output elements
    int total_elements = batch * num_heads * tokens_per_window * tokens_per_window;
    
    if (idx >= total_elements) return;
    
    // Decode which batch, head, and tokens this thread is responsible for
    int b = idx / (num_heads * tokens_per_window * tokens_per_window);
    int remaining = idx % (num_heads * tokens_per_window * tokens_per_window);
    
    int h = remaining / (tokens_per_window * tokens_per_window);
    remaining = remaining % (tokens_per_window * tokens_per_window);
    
    int token_i = remaining / tokens_per_window;  // row in attention matrix
    int token_j = remaining % tokens_per_window;  // column in attention matrix
    
    // Compute the dot product: sum over head_dim
    // output[b][h][token_i][token_j] = sum_d( query[b][h][token_i][d] * key[b][h][token_j][d] )
    
    float sum = 0.0f;
    
    // Find starting positions in the flat arrays
    // query is [batch, num_heads, tokens, head_dim]
    int query_base = ((b * num_heads + h) * tokens_per_window + token_i) * head_dim;
    
    // key is [batch, num_heads, tokens, head_dim]
    int key_base = ((b * num_heads + h) * tokens_per_window + token_j) * head_dim;
    
    // Dot product loop
    for (int d = 0; d < head_dim; d++) {
        sum += query[query_base + d] * key[key_base + d];
    }
    
    // Write result to output
    // output is [batch, num_heads, tokens_per_window, tokens_per_window]
    int output_idx = ((b * num_heads + h) * tokens_per_window + token_i) * tokens_per_window + token_j;
    output[output_idx] = sum;
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
    naive_attention_forward<<<blocks, threads_per_block>>>(
        query,
        key,
        output,
        batch,
        num_heads,
        tokens_per_window,
        head_dim
    );
}