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

# Progress bar
from tqdm import tqdm 

class Preprocessor:
    def __init__(self, input_path: str, output_path: str):
        self.data_dir = input_path
        self.pt_dir = output_path
    
    # Save the middle slice of volume of all modalities in a single figure -> save as PNG
    def SaveVolumeSlice(self, image: torch.Tensor, out_path: str):
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
        plt.savefig(os.path.join(out_path, 'preprocessed_all_modalities.png'), bbox_inches='tight', dpi=300)
        plt.close()

    # Remap BraTS labels: 0,1,2,4 -> 0,1,2,3
    def RemapBratsLabels(self, label_tensor: torch.Tensor) -> torch.Tensor:
        # label_tensor: shape (1, H, W, D)
        label_tensor = label_tensor.clone()  # avoid modifying original
        label_tensor[label_tensor == 4] = 3 # Map 4 -> 3
        return label_tensor

    # Ensure labels are int64 for training
    def CastLabelToLong(self, x: torch.Tensor) -> torch.Tensor:
        return x.long()
    
    def run(self):
        # Grab the path of all subjects from data_dir
        subjects = os.listdir(self.data_dir)

        # Create a list of dicts that contains paths for each test subject
        # Each dict element has {"image": [list of paths to the 4 modalities files] and "label" : [pash to segmentation file]}
        data_paths = []
        for subj in subjects:
            # Grab path
            subj_path = os.path.join(self.data_dir, subj)
            # Grab the 4 modalities (excluding seg)
            image_files = sorted([f for f in glob(os.path.join(subj_path, "*.nii.gz")) if "seg" not in f])
            # Grab seg file
            label_file = glob(os.path.join(subj_path, "*seg*.nii.gz"))[0]
            # Append to list
            data_paths.append({"image": image_files, "label": label_file})

        # Chain a series of deterministic preprocessing transforms to run in a sequential manner
        preprocess = Compose([
            LoadImaged(keys=["image", "label"]),                            # Load image & label
            EnsureChannelFirstd(keys=["image", "label"]),                   # Add channel dim
            Orientationd(keys=["image", "label"], axcodes="RAS"),           # Standard orientation
            Spacingd(keys=["image", "label"], pixdim=(1.0,1.0,1.0),         # Resample | bilinear for image, nearest for label
                     mode=("bilinear", "nearest")),                         
            ScaleIntensityRanged(                                           # Normalize intensities per modality
                keys=["image"], a_min=0, a_max=3000, 
                b_min=0.0, b_max=1.0, clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),   # Crop empty space
            Resized(keys=["image", "label"], spatial_size=(128,128,128),    # Resize to fixed volume | trilinear for image, nearest for labels
                    mode=("trilinear", "nearest")),                         
            LambdaD(keys=["label"], func=pre.RemapBratsLabels),             # Convert labels 0,1,2,4 → 0,1,2,3
            LambdaD(keys=["label"], func=pre.CastLabelToLong),              # Ensure labels are int64 for training
            EnsureTyped(keys=["image", "label"])                            # Convert to PyTorch tensors
        ])

        # Create Dataset + DataLoader
        dataset = Dataset(data=data_paths, transform=preprocess)
        loader = DataLoader(dataset, batch_size=1, num_workers=4)

        # Create output directory if it doesn't exist
        os.makedirs(self.pt_dir, exist_ok=True)

        for i, batch in enumerate(tqdm(loader)):                   # [batch, channel, H, W, D]
            image, label = batch["image"], batch["label"]  # Shapes: [1, 4, 128, 128, 128], [1, 1, 128, 128, 128]
            sample = {"image": image[0], "label": label[0]} # Shapes: [4, 128, 128, 128] and [1, 128, 128, 128]
            
            # Save visualization of the first subject's middle slice for all modalities
            if i == 0:
                image_save_path = "images"
                pre.SaveVolumeSlice(image, image_save_path)

                # Sanity check
                print("Image shape:", image[0].shape)
                print("Label shape:", label[0].shape)
                print("Label classes:", torch.unique(label))

            torch.save(sample, os.path.join(self.pt_dir, f"subject_{i:04d}.pt"))


if __name__ == "__main__":
    input_dir = "data\\BraTS2021_Training_Data"
    output_dir = "data\\preprocessed_tensors"

    # Preprocessor grabs the BraTS dataset from input_dir, performs preprocessing, and saves tensors to output_dir
    pre = Preprocessor(input_dir, output_dir)
    pre.run()