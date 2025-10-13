import os
from pathlib import Path
import shutil
import random

# Set random seed for reproducibility
random.seed(200)

# Define paths
source_dir = Path('data\\preprocessed_tensors')
base_dir = Path('data\\split_dataset')
train_dir = base_dir / 'train'
val_dir = base_dir / 'val'
test_dir = base_dir / 'test'

# Create directories if they don't exist
for dir_path in [train_dir, val_dir, test_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Get all .pt files
pt_files = list(source_dir.glob('*.pt'))
random.shuffle(pt_files)

# Calculate split sizes
total_files = len(pt_files)
train_size = int(0.8 * total_files)
val_size = int(0.1 * total_files)
# test_size will be the remaining files

# Split the files
train_files = pt_files[:train_size]
val_files = pt_files[train_size:train_size + val_size]
test_files = pt_files[train_size + val_size:]

# Function to copy files to destination
def copy_files(file_list, dest_dir):
    for file_path in file_list:
        shutil.copy2(file_path, dest_dir)
        print(f"Copied {file_path.name} to {dest_dir}")

# Copy files to respective directories
copy_files(train_files, train_dir)
copy_files(val_files, val_dir)
copy_files(test_files, test_dir)

print(f"\nDataset split complete:")
print(f"Train set: {len(train_files)} files")
print(f"Validation set: {len(val_files)} files")
print(f"Test set: {len(test_files)} files")