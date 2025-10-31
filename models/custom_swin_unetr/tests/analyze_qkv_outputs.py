import torch
import numpy as np
import matplotlib.pyplot as plt

def analyze_qkv_outputs(filepath='models/custom_swin_unetr/tests/results/qkv_test_outputs.pt'):
    # Load the saved outputs
    data = torch.load(filepath)
    monai_qkv = data['monai_qkv']
    cuda_qkv = data['cuda_qkv']
    
    # Calculate differences
    abs_diff = torch.abs(monai_qkv - cuda_qkv)
    rel_diff = abs_diff / (torch.abs(monai_qkv) + 1e-6)  # Add small epsilon to avoid division by zero
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Max absolute difference: {torch.max(abs_diff):.6e}")
    print(f"Mean absolute difference: {torch.mean(abs_diff):.6e}")
    print(f"Median absolute difference: {torch.median(abs_diff):.6e}")
    print(f"Max relative difference: {torch.max(rel_diff):.6e}")
    print(f"Mean relative difference: {torch.mean(rel_diff):.6e}")
    
    # Per-component analysis (Q, K, V)
    print("\n=== Per-Component Analysis ===")
    for i, name in enumerate(['Q', 'K', 'V']):
        component_diff = torch.abs(monai_qkv[i] - cuda_qkv[i])
        print(f"\n{name} Component:")
        print(f"Max difference: {torch.max(component_diff):.6e}")
        print(f"Mean difference: {torch.mean(component_diff):.6e}")
        print(f"Std difference: {torch.std(component_diff):.6e}")
    
    # Create histograms of differences
    plt.figure(figsize=(15, 5))
    
    # Absolute differences histogram
    plt.subplot(121)
    plt.hist(abs_diff.numpy().flatten(), bins=50)
    plt.title('Histogram of Absolute Differences')
    plt.xlabel('Absolute Difference')
    plt.ylabel('Count')
    plt.yscale('log')
    
    # Relative differences histogram
    plt.subplot(122)
    plt.hist(rel_diff.numpy().flatten(), bins=50)
    plt.title('Histogram of Relative Differences')
    plt.xlabel('Relative Difference')
    plt.ylabel('Count')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('models/custom_swin_unetr/tests/results/qkv_differences_histogram.png')
    plt.close()
    
    # Plot heatmap of differences for each component
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Heatmap of Absolute Differences per Component (Q, K, V)')
    
    for i, (name, ax) in enumerate(zip(['Q', 'K', 'V'], axes)):
        # Flatten all dimensions except the first for visualization
        diff_map = torch.mean(abs_diff[i], dim=(0, 1)).numpy()
        im = ax.imshow(diff_map, cmap='viridis')
        ax.set_title(f'{name} Component')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('models/custom_swin_unetr/tests/results/qkv_differences_heatmap.png')
    plt.close()
    
    # Value range analysis
    print("\n=== Value Range Analysis ===")
    print("MONAI QKV range:")
    print(f"Min: {torch.min(monai_qkv):.6f}, Max: {torch.max(monai_qkv):.6f}")
    print(f"Mean: {torch.mean(monai_qkv):.6f}, Std: {torch.std(monai_qkv):.6f}")
    
    print("\nCUDA QKV range:")
    print(f"Min: {torch.min(cuda_qkv):.6f}, Max: {torch.max(cuda_qkv):.6f}")
    print(f"Mean: {torch.mean(cuda_qkv):.6f}, Std: {torch.std(cuda_qkv):.6f}")

if __name__ == "__main__":
    analyze_qkv_outputs()