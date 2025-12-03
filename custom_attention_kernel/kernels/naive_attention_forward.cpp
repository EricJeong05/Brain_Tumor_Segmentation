#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declaration of the CUDA kernel launcher
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
    int threads_per_block);

// Python-callable wrapper function
torch::Tensor naive_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    int batch,
    int num_heads,
    int tokens_per_window,
    int head_dim)
{
    // Allocate output tensors
    auto options = torch::TensorOptions()
        .dtype(query.dtype())
        .device(query.device());
    
    // attn_probs: [batch, num_heads, tokens, tokens] - attention probabilities after softmax
    auto attn_probs = torch::zeros({batch, num_heads, tokens_per_window, tokens_per_window}, options);
    
    // output: [batch, num_heads, tokens, head_dim] - final attention output (attn @ v)
    auto output = torch::zeros({batch, num_heads, tokens_per_window, head_dim}, options);
    
    // Calculate grid dimensions
    int total_elements = batch * num_heads * tokens_per_window * tokens_per_window;
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    launch_naive_attention_forward(
        query.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        attn_probs.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        num_heads,
        tokens_per_window,
        head_dim,
        blocks,
        threads_per_block
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_attention_forward", &naive_attention_forward, "Naive attention forward (CUDA)");
}
