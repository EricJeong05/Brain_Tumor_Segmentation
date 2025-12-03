# Understanding the Attention CUDA Kernel: A Beginner's Guide

## Table of Contents
1. [Introduction](#introduction)
2. [What Problem Are We Solving?](#what-problem-are-we-solving)
3. [The Mathematics Behind Attention](#the-mathematics-behind-attention)
4. [Breaking Down the CUDA Kernel](#breaking-down-the-cuda-kernel)
5. [Shared Memory and Tiling Strategy](#shared-memory-and-tiling-strategy)
6. [Step-by-Step Execution Example](#step-by-step-execution-example)
7. [Performance Optimizations](#performance-optimizations)
8. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)

---

## Introduction

This document explains a **custom CUDA kernel** that computes the attention mechanism used in transformer models, specifically the Swin Transformer for brain tumor segmentation. The kernel implements the matrix multiplication: **attention = Q @ K^T** (query multiplied by key transpose).

**Target Audience**: Programmers with basic understanding of:
- Matrix multiplication
- Basic CUDA concepts (threads, blocks, grids)
- C/C++ programming

**What You'll Learn**:
- How attention computation works mathematically
- How to use shared memory to optimize GPU kernels
- How tiling improves performance on modern GPUs
- How to think about parallelizing matrix operations

---

## What Problem Are We Solving?

### The Attention Mechanism

In transformer models (like BERT, GPT, Vision Transformers), the **attention mechanism** determines how much each token (piece of data) should "pay attention" to every other token. For medical image segmentation, this helps the model understand spatial relationships between different parts of the brain scan.

### The Computation

Given:
- **Q (Query)**: Shape `[batch, heads, tokens, dim]` - "What am I looking for?"
- **K (Key)**: Shape `[batch, heads, tokens, dim]` - "What information do I have?"

We need to compute:
- **Attention Scores**: Shape `[batch, heads, tokens, tokens]` = Q @ K^T

### Why Use CUDA?

PyTorch can do this computation automatically, but:
1. **Learning opportunity**: Understand GPU programming deeply
2. **Performance**: Custom kernels can sometimes outperform general solutions
3. **Control**: Full control over memory access patterns and optimizations

---

## The Mathematics Behind Attention

### Simple Example

Let's use tiny matrices to understand what we're computing:

```
Q = [batch=1, heads=1, tokens=3, dim=2]
    Token 0: [1.0, 2.0]
    Token 1: [3.0, 4.0]
    Token 2: [5.0, 6.0]

K = [batch=1, heads=1, tokens=3, dim=2]
    Token 0: [0.5, 1.5]
    Token 1: [2.5, 3.5]
    Token 2: [4.5, 5.5]
```

### Computing K^T (Key Transpose)

First, we transpose K from `[3 tokens × 2 dim]` to `[2 dim × 3 tokens]`:

```
K^T = [2 dim × 3 tokens]
      [0.5  2.5  4.5]
      [1.5  3.5  5.5]
```

### Computing Q @ K^T

Now multiply Q `[3×2]` by K^T `[2×3]` to get Attention `[3×3]`:

```
Attention[0,0] = Q[0] · K[0] = (1.0×0.5) + (2.0×1.5) = 0.5 + 3.0 = 3.5
Attention[0,1] = Q[0] · K[1] = (1.0×2.5) + (2.0×3.5) = 2.5 + 7.0 = 9.5
Attention[0,2] = Q[0] · K[2] = (1.0×4.5) + (2.0×5.5) = 4.5 + 11.0 = 15.5

Attention[1,0] = Q[1] · K[0] = (3.0×0.5) + (4.0×1.5) = 1.5 + 6.0 = 7.5
Attention[1,1] = Q[1] · K[1] = (3.0×2.5) + (4.0×3.5) = 7.5 + 14.0 = 21.5
... and so on
```

**Result** (Attention matrix):
```
Attention = [3×3]
    [3.5   9.5   15.5]
    [7.5   21.5  35.5]
    [11.5  33.5  55.5]
```

Each cell `Attention[i,j]` represents how much token `i` attends to token `j`.

---

## Breaking Down the CUDA Kernel

### Kernel Overview

```cuda
__global__ void naive_attention_forward(
    const float* __restrict__ query,   // Input: Q matrix
    const float* __restrict__ key,     // Input: K matrix
    float* __restrict__ output,        // Output: Attention matrix
    int batch, int num_heads,
    int tokens_per_window,
    int head_dim)
```

### Key Components

#### 1. **Thread Organization** (Lines 18-34)

```cuda
// Thread position within a block
int tx = threadIdx.x;  // Thread's x-coordinate (0 to 31)
int ty = threadIdx.y;  // Thread's y-coordinate (0 to 31)

// Block position in grid
int block_row = blockIdx.y;  // Which row of tiles (vertical)
int block_col = blockIdx.x;  // Which column of tiles (horizontal)

// Which batch and head are we processing?
int bh = blockIdx.z;
int b = bh / num_heads;      // Batch index
int h = bh % num_heads;      // Head index

// Global position in output matrix
int row = block_row * TILE_SIZE + ty;  // Output row (0 to tokens-1)
int col = block_col * TILE_SIZE + tx;  // Output column (0 to tokens-1)
```

**Visualization**: If we have 64 tokens with TILE_SIZE=32:

```
Grid Layout (looking down from above):
┌─────────────┬─────────────┐
│  Block(0,0) │  Block(1,0) │  ← block_row = 0
│  32×32 tile │  32×32 tile │
├─────────────┼─────────────┤
│  Block(0,1) │  Block(1,1) │  ← block_row = 1
│  32×32 tile │  32×32 tile │
└─────────────┴─────────────┘
   block_col=0   block_col=1

Each block contains 32×32 = 1,024 threads
Each thread computes ONE output element
```

#### 2. **Shared Memory Tiles** (Lines 15-16)

```cuda
__shared__ float tile_query[TILE_SIZE][TILE_SIZE];  // 32×32 floats = 4 KB
__shared__ float tile_key[TILE_SIZE][TILE_SIZE];    // 32×32 floats = 4 KB
```

**What is Shared Memory?**
- **Fast memory** shared by all threads in a block (~100× faster than global memory)
- **Limited size**: 128 KB per streaming multiprocessor on RTX 4060Ti
- **Purpose**: Cache frequently accessed data to avoid slow global memory reads

**Why Use Tiles?**
Instead of each thread reading from slow global memory repeatedly, we:
1. **Collaboratively load** a tile of data into shared memory (all threads help)
2. **Reuse** that data many times (fast shared memory access)
3. **Move to next tile** and repeat

#### 3. **The Tiling Loop** (Lines 45-80)

This is where the magic happens! Let's break it down:

```cuda
// How many tiles do we need to cover the head_dim dimension?
int num_tiles = (head_dim + TILE_SIZE - 1) / TILE_SIZE;

// Example: If head_dim=64 and TILE_SIZE=32, we need 2 tiles
```

**For each tile iteration:**

##### Step A: Load Query Tile (Lines 48-55)
```cuda
int dim_idx = tile * TILE_SIZE + tx;

if (row < tokens_per_window && dim_idx < head_dim) {
    int query_idx = bh_offset + row * head_dim + dim_idx;
    tile_query[ty][tx] = query[query_idx];
} else {
    tile_query[ty][tx] = 0.0f;  // Padding for boundary cases
}
```

**What's happening:**
- Thread `(tx, ty)` loads one element from global Q matrix
- Stores it at `tile_query[ty][tx]` in shared memory
- If out of bounds, pad with zero

**Example** (head_dim=64, tile=0):
```
Thread(0,0) loads Q[row=0, dim=0]  → tile_query[0][0]
Thread(1,0) loads Q[row=0, dim=1]  → tile_query[0][1]
...
Thread(31,0) loads Q[row=0, dim=31] → tile_query[0][31]
Thread(0,1) loads Q[row=1, dim=0]  → tile_query[1][0]
...
All 1,024 threads load their assigned element in parallel!
```

##### Step B: Load Key Tile (Lines 58-65)
```cuda
dim_idx = tile * TILE_SIZE + ty;  // Note: using 'ty' for dimension

if (col < tokens_per_window && dim_idx < head_dim) {
    int key_idx = bh_offset + col * head_dim + dim_idx;
    tile_key[ty][tx] = key[key_idx];
} else {
    tile_key[ty][tx] = 0.0f;
}
```

**Why different indexing?**
- We're computing K^T (transpose), so we load K in a transposed pattern
- `tile_key[ty][tx]` stores `K[col, dim]` which gives us K^T[dim, col]

##### Step C: Synchronize (Line 68)
```cuda
__syncthreads();  // Wait for all threads to finish loading
```

**Critical!** Ensures all threads have loaded their data before anyone starts computing.

##### Step D: Compute Partial Dot Product (Lines 71-75)
```cuda
#pragma unroll
for (int k = 0; k < TILE_SIZE; k++) {
    sum += tile_query[ty][k] * tile_key[k][tx];
}
```

**What's happening:**
- Thread `(tx, ty)` computes a **partial** dot product
- Multiplies elements from its row in `tile_query` with its column in `tile_key`
- Accumulates into `sum`

**Example** (simplified 3×3 tiles for clarity):
```
Thread(0,0) computes:
sum += tile_query[0][0] * tile_key[0][0]  // k=0
sum += tile_query[0][1] * tile_key[1][0]  // k=1
sum += tile_query[0][2] * tile_key[2][0]  // k=2

Thread(1,0) computes:
sum += tile_query[0][0] * tile_key[0][1]  // k=0
sum += tile_query[0][1] * tile_key[1][1]  // k=1
sum += tile_query[0][2] * tile_key[2][1]  // k=2
```

##### Step E: Synchronize Again (Line 78)
```cuda
__syncthreads();  // Wait before loading next tile
```

Ensures no thread overwrites shared memory while others are still computing.

#### 4. **Write Result** (Lines 83-87)

```cuda
if (row < tokens_per_window && col < tokens_per_window) {
    int output_idx = bh_offset / head_dim * tokens_per_window 
                     + row * tokens_per_window + col;
    output[output_idx] = sum;
}
```

After all tiles processed, write the final accumulated `sum` to global memory.

---

## Shared Memory and Tiling Strategy

### Why Tiling?

**Problem without tiling:**
```
For each output element Attention[i,j]:
    For k = 0 to head_dim:
        Read Q[i,k] from global memory    ← SLOW! (100+ cycles)
        Read K[j,k] from global memory    ← SLOW!
        Multiply and accumulate
```

If `head_dim = 64`, each output element requires **128 global memory reads** (64 for Q, 64 for K).

**Solution with tiling:**
```
For each tile along head_dim:
    Collaboratively load Q tile → shared memory   ← One slow read per thread
    Collaboratively load K tile → shared memory   ← One slow read per thread
    
    For k = 0 to TILE_SIZE:
        Read from shared memory    ← FAST! (~5 cycles)
        Multiply and accumulate
```

### Memory Access Pattern

**Visual representation** (computing 4×4 output with 2×2 tiles):

```
Iteration 0: Load Q[:,0:2] and K[:,0:2]
┌───────┐     ┌───────┐
│ Q tile│  @  │K^T til│  →  Partial sum
│  (2)  │     │  (2)  │
└───────┘     └───────┘

Iteration 1: Load Q[:,2:4] and K[:,2:4]
┌───────┐     ┌───────┐
│ Q tile│  @  │K^T til│  →  Add to sum
│  (2)  │     │  (2)  │
└───────┘     └───────┘

Final result = sum from all iterations
```

### Data Reuse

**Key insight**: Each tile element is reused TILE_SIZE times!

```
tile_query[0][5] is used by:
  - Thread(0,0), Thread(1,0), ..., Thread(31,0)  [32 threads]

That's 32 reuses from shared memory vs. 32 global memory reads!
```

**Speedup calculation:**
- Global memory: ~400 cycles latency
- Shared memory: ~5 cycles latency
- **Reuse factor**: 32× per tile element
- **Effective speedup**: ~10-20× for memory-bound operations

---

## Step-by-Step Execution Example

Let's trace a concrete example with small dimensions:

### Setup
```
batch = 1
num_heads = 1
tokens = 4
head_dim = 4
TILE_SIZE = 2

Q = [4 tokens × 4 dim]
    [1, 2, 3, 4]
    [5, 6, 7, 8]
    [2, 3, 4, 5]
    [6, 7, 8, 9]

K = [4 tokens × 4 dim]
    [1, 1, 1, 1]
    [2, 2, 2, 2]
    [3, 3, 3, 3]
    [4, 4, 4, 4]
```

### Grid/Block Configuration
```
blockDim = (2, 2)        # 2×2 threads per block = 4 threads
gridDim = (2, 2, 1)      # 2×2 blocks = 4 blocks total
```

### Block(0,0) - Computing Output[0:2, 0:2]

**Thread mapping:**
```
Thread(0,0) → Output[0,0]
Thread(1,0) → Output[0,1]
Thread(0,1) → Output[1,0]
Thread(1,1) → Output[1,1]
```

#### Tile Iteration 0 (dimensions 0-1)

**Load Query Tile:**
```
Thread(0,0): tile_query[0][0] = Q[0,0] = 1
Thread(1,0): tile_query[0][1] = Q[0,1] = 2
Thread(0,1): tile_query[1][0] = Q[1,0] = 5
Thread(1,1): tile_query[1][1] = Q[1,1] = 6

Result:
tile_query = [[1, 2],
              [5, 6]]
```

**Load Key Tile (transposed):**
```
Thread(0,0): tile_key[0][0] = K[0,0] = 1
Thread(1,0): tile_key[0][1] = K[1,0] = 2
Thread(0,1): tile_key[1][0] = K[0,1] = 1
Thread(1,1): tile_key[1][1] = K[1,1] = 2

Result (this represents K^T):
tile_key = [[1, 2],
            [1, 2]]
```

**Compute (Thread 0,0 computing Output[0,0]):**
```
sum = 0
k=0: sum += tile_query[0][0] * tile_key[0][0] = 1 × 1 = 1
k=1: sum += tile_query[0][1] * tile_key[1][0] = 2 × 1 = 2
sum = 3  (partial result)
```

#### Tile Iteration 1 (dimensions 2-3)

**Load Query Tile:**
```
tile_query = [[3, 4],
              [7, 8]]
```

**Load Key Tile:**
```
tile_key = [[1, 2],
            [1, 2]]
```

**Compute (Thread 0,0 continuing Output[0,0]):**
```
k=0: sum += 3 × 1 = 3
k=1: sum += 4 × 1 = 4
sum = 3 + 3 + 4 = 10  (final result)
```

**Write:** `Output[0,0] = 10`

**Verification** (manual calculation):
```
Output[0,0] = Q[0] · K[0]
            = [1,2,3,4] · [1,1,1,1]
            = 1+2+3+4
            = 10 ✓
```

### All Blocks Execute in Parallel

```
Block(0,0) computes Output[0:2, 0:2]
Block(1,0) computes Output[0:2, 2:4]
Block(0,1) computes Output[2:4, 0:2]
Block(1,1) computes Output[2:4, 2:4]

All happening simultaneously on different SMs!
```

---

## Performance Optimizations

### 1. **The `__restrict__` Keyword**

```cuda
const float* __restrict__ query
```

**What it does:**
- Tells compiler: "This pointer doesn't alias with any other pointer"
- Allows aggressive optimizations (vectorization, reordering)

**Example:**
```cuda
// Without __restrict__: Compiler must be conservative
float a = *ptr1;
float b = *ptr2;  // Could ptr2 point to same location as ptr1?
float c = a + b;  // Must read ptr1 again after ptr2 write

// With __restrict__: Compiler knows they don't overlap
float a = *ptr1;  // Can be cached in register
float b = *ptr2;
float c = a + b;  // No need to re-read ptr1
```

### 2. **`#pragma unroll`**

```cuda
#pragma unroll
for (int k = 0; k < TILE_SIZE; k++) {
    sum += tile_query[ty][k] * tile_key[k][tx];
}
```

**What it does:**
- Unrolls the loop at compile time
- Reduces loop overhead (no counter increment, no branch)

**Before (loop):**
```assembly
loop:
    load tile_query[ty][k]
    load tile_key[k][tx]
    multiply and add
    increment k
    compare k < TILE_SIZE
    branch if not done
```

**After (unrolled for TILE_SIZE=4):**
```assembly
load tile_query[ty][0]; load tile_key[0][tx]; multiply-add
load tile_query[ty][1]; load tile_key[1][tx]; multiply-add
load tile_query[ty][2]; load tile_key[2][tx]; multiply-add
load tile_query[ty][3]; load tile_key[3][tx]; multiply-add
```

### 3. **Boundary Checking**

```cuda
if (row < tokens_per_window && dim_idx < head_dim) {
    // Load data
} else {
    tile_query[ty][tx] = 0.0f;  // Pad with zeros
}
```

**Why needed:**
- Tile dimensions may not evenly divide matrix dimensions
- Example: 50 tokens with TILE_SIZE=32 → last tile has 18 real elements + 14 padding

### 4. **Coalesced Memory Access**

**Good pattern** (consecutive threads access consecutive memory):
```cuda
// Thread 0 reads address 0
// Thread 1 reads address 4 (next float)
// Thread 2 reads address 8
// → GPU combines into one memory transaction!
```

**Bad pattern** (scattered reads):
```cuda
// Thread 0 reads address 0
// Thread 1 reads address 1000
// Thread 2 reads address 2000
// → Requires multiple separate transactions (slow)
```

Our kernel uses **strided access** which is reasonably efficient on modern GPUs.

---

## Common Pitfalls and Solutions

### Pitfall 1: Missing `__syncthreads()`

**Problem:**
```cuda
// Load tiles
tile_query[ty][tx] = ...;
tile_key[ty][tx] = ...;
// ❌ MISSING SYNC!

// Compute immediately
for (int k = 0; k < TILE_SIZE; k++) {
    sum += tile_query[ty][k] * tile_key[k][tx];
}
```

**What happens:**
- Fast threads start computing before slow threads finish loading
- Read garbage data or incomplete tiles
- **Result**: Wrong answers, hard-to-debug race conditions

**Solution:** Always sync after collaborative loads!

### Pitfall 2: Shared Memory Bank Conflicts

**Problem** (accessing same bank):
```cuda
// All threads access column 0
float val = tile[threadIdx.y][0];  // Bank conflict!
```

**Solution** (spread across banks):
```cuda
// Each thread accesses different column
float val = tile[threadIdx.y][threadIdx.x];  // No conflict
```

Our kernel naturally avoids this in the inner loop.

### Pitfall 3: Incorrect Index Calculations

**Problem:**
```cuda
// Wrong! Doesn't account for batch/head structure
int query_idx = row * head_dim + dim_idx;
```

**Solution:**
```cuda
// Correct! Includes batch and head offsets
int bh_offset = (b * num_heads + h) * tokens_per_window * head_dim;
int query_idx = bh_offset + row * head_dim + dim_idx;
```

### Pitfall 4: Forgetting Transpose for Key

**Problem:**
```cuda
// Loading K the same way as Q
tile_key[ty][tx] = key[row * head_dim + dim_idx];
```

**Solution:**
```cuda
// Loading K in transposed pattern for K^T
tile_key[ty][tx] = key[col * head_dim + dim_idx];
```

---

## Performance Metrics

### Expected Performance (RTX 4060Ti)

For typical transformer attention:
- **Input**: batch=2, heads=12, tokens=196, head_dim=64
- **Output**: [2, 12, 196, 196] = ~900K elements

**Theoretical analysis:**
```
Work per output element:
  - 64 multiply-adds (dot product)
  - ~900K total elements
  - ~58M operations

RTX 4060Ti specs:
  - 4352 CUDA cores @ 2.5 GHz
  - ~22 TFLOPS FP32 (theoretical peak)

Memory bandwidth:
  - 288 GB/s
  - Attention is memory-bound (low arithmetic intensity)
```

**Realistic expectations:**
- **Custom kernel**: 0.1-0.5 ms for small problems
- **PyTorch cuBLAS**: Highly optimized, may be faster for large matrices
- **Speedup**: Depends on problem size and memory access patterns

### Benchmark Command

```bash
python attention_comparison_test.py --benchmark --iterations 100 --warmup 10
```

---

## Further Optimizations (Advanced)

If you want to push performance further:

### 1. **Register Blocking**
```cuda
// Compute multiple outputs per thread
float sum[4][4];  // Each thread computes 4×4 output tile
```

### 2. **Warp-Level Primitives**
```cuda
// Use warp shuffle for reduction
sum = __shfl_down_sync(0xffffffff, sum, offset);
```

### 3. **Tensor Cores** (requires FP16)
```cuda
// Use wmma API for mixed-precision matrix multiply
wmma::fragment<...> a, b, c;
wmma::mma_sync(c, a, b, c);
```

### 4. **Asynchronous Copy**
```cuda
// Copy global → shared asynchronously (Ampere+)
__pipeline_memcpy_async(&tile[ty][tx], &global[idx], sizeof(float));
```

---

## Summary

This CUDA kernel implements **tiled matrix multiplication** for attention computation:

1. **Divides work** into tiles that fit in fast shared memory
2. **Loads tiles collaboratively** (all threads help load data)
3. **Reuses data** from shared memory many times (key optimization)
4. **Computes partial results** tile-by-tile
5. **Writes final result** back to global memory

**Key takeaways:**
- ✅ Shared memory is 100× faster than global memory
- ✅ Tiling enables data reuse and reduces memory traffic
- ✅ Thread synchronization is critical for correctness
- ✅ Memory access patterns greatly impact performance
- ✅ Modern GPUs favor warp-aligned operations (32 threads)

**Next steps:**
1. Run the benchmark to measure actual performance
2. Try different TILE_SIZE values (16, 32, 64)
3. Profile with Nsight Compute to find bottlenecks
4. Experiment with advanced optimizations

---

## Additional Resources

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Profiler](https://developer.nvidia.com/nsight-compute)
- [Matrix Multiplication in CUDA](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)

**Questions?** Review the code with this guide, and experiment with modifications to deepen your understanding!
