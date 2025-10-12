# Real time GPU-Accelerated 3D Vision Transformer for Brain Tumor Segmentation
This is the next stage in my GPU/ML learning journey (see [Real-Time-Webcam-Image-Classification](https://github.com/EricJeong05/Real-Time-Webcam-Image-Classification) for my first project!) where I try and tackle more complex subjects (like going from 2D -> 3D images) and go through the entire ML pipeline from data preprocessing -> model training -> inference -> model performance optimization and GPU acceleration!  

## 1 . Visualize the BraTs data to first understand what the data looks like
I used the 2021 BRaTS dataset for this project ([Kaggle link](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)). The first step I wanted to do was just see what this dataset even looked like since I've never worked with MRI data before. So the view_interactive_4pane_brats.py script loads all 4 of the modalities (T1, T1CE, T2, FLAIR) into one plot and overlays the segmentation masks on top of each to show where the tumor is. 

This script is interactive so that when you use your mouse scroll wheel to scroll up & down, it goes through the slices (z-axis) of the MRI volume.

**<ins>Here's a slice of the volume visualized:</ins>**

![orig_all_modalities](/images/orig_all_modalities.png)

## 2. Preprocess the data
The next step is to preprocess the dataset to get it ready for feeding it into the model. For this project I'm preprocessing it so that it can be both fed into a UNet model (CNN-based) and a SwinUNETR model (transformer-based)

To allow support for both these models and ease of re-use between models, I've opted to go for a preprocessor script that takes the entire BRaTS dataset and does all of the deterministic preprocessing "offline" and stores it in a tensor (.pt) format for easy and quick loading during training.

**<ins>The preprocessing steps include:</ins>**

1. Add channel dimension
2. Reorient volume to standard axis convention (RAS)
3. Resample to isotropic voxel spacing (e.g., 1mm × 1mm × 1mm) using linear interpolation for images and nearest-neighbor for labels
4. Normalize intensities from 0-1 per modality
5. Foreground crop to remove empty black space and resize all volumes to 128x128x128
6. Convert labels 0,1,2,4 → 0,1,2,3 for ease of use
7. Ensure labels are int64 (not float64) for training
8. Save as PyTorch tensors (.pt)

**<ins>After preprocessing, the volume looks like such:</ins>**

![preprocessed_all_modalities](/images/preprocessed_all_modalities.png)

3. Build a baseline 3D U-Net model (CNN) and grab baseline measurements


4. Build a transformer-based 3D segmentation model (UNETR or SwinUNETR) and compare with U-Net
5. Optimize components with GPU & grab performance improvements
