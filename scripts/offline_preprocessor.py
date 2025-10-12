import os
import torch
import matplotlib.pyplot as plt
from glob import glob
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, Resized,
    Compose, EnsureTyped, LambdaD
)
from monai.data import Dataset, DataLoader
from tqdm import tqdm # Progress bar

# ---- 1. Define custom label remapping transform ----
def remap_brats_labels(label_tensor: torch.Tensor) -> torch.Tensor:
    # label_tensor: shape (1, H, W, D)
    label_tensor = label_tensor.clone()  # avoid modifying original
    label_tensor[label_tensor == 4] = 3 # Map 4 -> 3
    return label_tensor

def cast_label_to_int(x: torch.Tensor) -> torch.Tensor:
    return x.long()

# ---- 2. File discovery (BraTS format: 4 modalities + segmentation) ----
#data_dir = "data\\test"
data_dir = "data\\BraTS2021_Training_Data"
subjects = os.listdir(data_dir)

# Create list of dicts that contains paths for each test subject
data_dicts = []
for subj in subjects:
    # Grab path
    subj_path = os.path.join(data_dir, subj)
    # Grab 4 modalities (excluding seg)
    image_files = sorted([f for f in glob(os.path.join(subj_path, "*.nii.gz")) if "seg" not in f])
    # Grab seg file
    label_file = glob(os.path.join(subj_path, "*seg*.nii.gz"))[0]
    # Append to list
    data_dicts.append({"image": image_files, "label": label_file})

# ---- 3. Deterministic preprocessing transforms ----
preprocess = Compose([
    LoadImaged(keys=["image", "label"]),                    # Load NIfTI
    EnsureChannelFirstd(keys=["image", "label"]),           # Add channel dim
    Orientationd(keys=["image", "label"], axcodes="RAS"),   # Standard orientation
    Spacingd(keys=["image", "label"], pixdim=(1.0,1.0,1.0), mode=("bilinear", "nearest")),  # Resample | bilinear for image, nearest for label
    ScaleIntensityRanged(                                   # Normalize intensities per modality
        keys=["image"], a_min=0, a_max=3000, 
        b_min=0.0, b_max=1.0, clip=True
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"), # Crop empty space
    Resized(keys=["image", "label"], spatial_size=(128,128,128), mode=("trilinear", "nearest")), # Resize to fixed volume | trilinear for image, nearest for labels
    LambdaD(keys=["label"], func=remap_brats_labels),       # Convert labels 0,1,2,4 → 0,1,2,3
    LambdaD(keys=["label"], func=cast_label_to_int),        # Ensure labels are int64 for training
    EnsureTyped(keys=["image", "label"])                    # Convert to PyTorch tensors
])

# ---- 4. Dataset + DataLoader (no augmentation here) ----
dataset = Dataset(data=data_dicts, transform=preprocess)
loader = DataLoader(dataset, batch_size=1, num_workers=4)

# ---- 5. Save preprocessed tensors in separate directory----
if __name__ == "__main__":
    #out_dir = "data\\preprocessed_pt\\test"
    out_dir = "data\\preprocessed_pt"
    os.makedirs(out_dir, exist_ok=True)

    for i, batch in enumerate(tqdm(loader)):                   # [batch, channel, H, W, D]
        image, label = batch["image"], batch["label"]  # Shapes: [1, 4, 128, 128, 128], [1, 1, 128, 128, 128]
        sample = {"image": image[0], "label": label[0]} # Shapes: [4, 128, 128, 128] and [1, 128, 128, 128]
        
        # Save visualization of the first subject's middle slice for all modalities
        if i == 0:
            modality_names = ['T1', 'T1ce', 'T2', 'FLAIR']
            
            # Create a figure with 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            
            # Plot each modality
            for mod_idx, (modality, ax) in enumerate(zip(modality_names, axes.flat)):
                # Get middle slice for each modality
                middle_slice = image[0, mod_idx, :, :, 64].cpu().numpy()  # Take slice 64 out of 128
                
                im = ax.imshow(middle_slice, cmap='gray')
                ax.set_title(f'Preprocessed {modality} Middle Slice')
                ax.axis('off')
                plt.colorbar(im, ax=ax, label='Intensity')
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'preprocessed_all_modalities.png'), bbox_inches='tight', dpi=300)
            plt.close()
            
        torch.save(sample, os.path.join(out_dir, f"subject_{i:04d}.pt"))
        #print("Image shape:", image[0].shape)
        #print("Label shape:", label[0].shape)
        #print("Label classes:", torch.unique(label))