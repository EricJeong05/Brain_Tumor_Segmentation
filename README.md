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

## 3. Train UNet Model for Baseline DICE
Now once all the preprocessing is complete and we have all the .pt tensors saved, I'm training a basic 3D U-Net model (CNN) model to grab the baseline to compare the transformer-based SwinUNETR model to and just purely out of my own interest and curiosity. It was pretty cool to learn about how the UNet architecture just works. Shown below:

![unet_diagram](/images/UNet_diagram.png)

During each training step, we perform real-time augmentations to the input tensor to improve training performance. These augmentations include:

1. Random flips (axis-wise)
2. Random rotations (90deg)
3. Random gaussian noise

We train the UNet for 200 epochs and save the model with the best DICE score since it's not guarenteed that the most recently trained model provides the best DICE score. 

State of the art UNet models can reach over DICE scores of 90%+, but as this is a very basic/simple UNet model I'm using, my model reaches:

- **DICE Score = 0.7633**
- **Training Loss = 0.2496**
- **Validation Loss = 0.2990**

We plot the training and validation loss over all the epochs to see how the model faired during training:

![unet_diagram](/models/unet/results/training_plots.png)

**Total time for training: 447.19 minutes**

Then we use the best performing model, load it, and run it through the test dataset and see how it performs. As shown in the DICE score and Loss below, it does perform a bit worse compared to the training data, but this is expected and it being very close to the training scores, this is pretty good!

- **DICE Score = 0.7616**
- **Test Loss = 0.2947**

Below is one of the predictions it made compared to the ground truth:

![test_prediction_overlay](/models/unet/results/test_prediction_overlay.png)


## 4. Train SwinUNETR model and compare with UNet
## 5. GPU Acceleration
