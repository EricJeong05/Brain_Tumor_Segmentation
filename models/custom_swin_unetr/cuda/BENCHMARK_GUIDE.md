# Attention Kernel Benchmark Guide

## Quick Usage

```powershell
# Basic test (correctness only)
python attention_comparison_test.py

# Benchmark with defaults, single size (10000 iterations, 10 warmup)
python attention_comparison_test.py --benchmark

# More iterations for finer accuracy
python attention_comparison_test.py --benchmark --iterations 50000

# Fast benchmark (fewer iterations)
python attention_comparison_test.py --benchmark --iterations 5000 --warmup 5

# Test multiple sizes
python attention_comparison_test.py --benchmark-sizes

# Custom multi-size benchmark
python attention_comparison_test.py --benchmark-sizes --iterations 50000
```

## What Gets Measured

The benchmark measures:
- **PyTorch GPU (cuBLAS)**:

```python
q_scaled @ k.transpose(-2, -1)
```


- **CUDA GPU**: naive_attention_foward.cu

For each method:
- Mean execution time (milliseconds)
- Standard deviation
- Min/Max times
- Speedup ratio (PyTorch / CUDA)
- Correctness verification (max difference)

## Benchmark Features

### Warmup Runs
- Default: 10 warmup iterations
- Ensures GPU is fully initialized
- Warmup times are not included in measurements

### Timing Iterations
- Default: 10000 timing iterations
- Averaged to get stable measurements
- More iterations = more accurate but slower

### CUDA Synchronization
- `torch.cuda.synchronize()` before and after timing
- Ensures accurate GPU timing (no async execution)

### Multi-Size Benchmark

Tests configurations:
1. **Small**: batch=1, heads=4, tokens=8, dim=16
2. **Medium**: batch=2, heads=6, tokens=49, dim=32 (7×7 window)
3. **Large**: batch=4, heads=12, tokens=343, dim=32 (7×7×7 window)
4. **Very Large**: batch=8, heads=24, tokens=343, dim=32

### Tips for Best Results

1. **Close other GPU applications** to reduce variance
2. **Run multiple times** to check consistency
3. **Larger problems** show better GPU speedup
4. **Small problems** may be CPU-limited due to kernel launch overhead
5. **Check correctness first** before benchmarking

## Test Results
````
================================================================================
BENCHMARK SUMMARY: naive_attention_forward (initial implementation)
================================================================================
Config                         PyTorch (ms)    CUDA (ms)       Speedup
--------------------------------------------------------------------------------
B1_H4_T8_D16                   0.0335          0.0276          1.21x (negligible)
B2_H6_T49_D32                  0.0355          0.0357         -1.01x (negligible)
B4_H12_T343_D32                0.1225          2.2463         -18.34x
B8_H24_T343_D32                0.4983          9.0755         -18.21x


================================================================================
BENCHMARK SUMMARY: naive_attention_forward (tiled memory) 
================================================================================
Config                         PyTorch (ms)    CUDA (ms)       Speedup
--------------------------------------------------------------------------------
B1_H4_T8_D16                   0.0305          0.0254          1.20x (negligible)
B2_H6_T49_D32                  0.0344          0.0276          1.25x (negligible)
B4_H12_T343_D32                0.1192          0.3999         -3.35x
B8_H24_T343_D32                0.4765          1.6728         -3.51x
````