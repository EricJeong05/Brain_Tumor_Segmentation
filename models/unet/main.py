import os
import time
import csv
import torch
import torch.nn.functional as F
from glob import glob
from datetime import datetime
from monai.data import DataLoader
from tqdm import tqdm
from monai.transforms import (
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    Compose
)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
import numpy as np

# Define custom dataset that loads torch tensors 
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load the .pt file
        sample = torch.load(self.files[idx], weights_only = False)

        # Grab image and label tensors
        img, label = sample["image"], sample["label"]

        # Apply transforms if any
        sample = {"image": img, "label": label}
        if self.transform:
            sample = self.transform(sample)

        return sample

class UnetModel:
    def __init__(self, train_path: str, val_path: str, results_path: str):
        #  Get list of preprocessed training and validation .pt files 
        self.train_files = glob(os.path.join(train_path, "*.pt"))
        self.val_files = glob(os.path.join(val_path, "*.pt"))
        self.results_dir = results_path

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}") 

        # Define UNet model 
        self.model = UNet(
            spatial_dims=3,                     # 3D volumes
            in_channels=4,                      # BraTS modalities: flair, t1, t1ce, t2
            out_channels=4,                     # classes: background, edema, non-enhancing, enhancing
            channels=(16, 32, 64, 128, 256),    # Channels at each level in the encoder
            strides=(2, 2, 2, 2),               # Downsample by factor of 2 at each layer
            num_res_units=2,                    # Number of residual units (Conv+RELU) at each layer   
        ).to(self.device)   

    def train(self, epochs: int = 50):
        # Real time augmentations for training (include rotations, flips, noise)
        train_transforms = Compose([
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1)
        ])

        # Create datasets (validation = no augmentation)
        train_ds = TorchDataset(self.train_files, transform=train_transforms)
        val_ds = TorchDataset(self.val_files, transform=None)

        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)
        
        # Use ADAM with learning rate 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # Use Dice + CrossEntropy loss
        loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        # Use Dice metric for evaluation
        dice_metric = DiceMetric(include_background=False, reduction="mean")

        # Create results directory if it doesn't exist
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialize CSV logging
        csv_filename = os.path.join(self.results_dir, f"training_results.csv")
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Dice Score', 'Time (s)'])

        best_dice = 0.0
        start_time = time.time()

        # Training loop (# of epochs)
        for epoch in range(epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            for batch in tqdm(train_loader, desc="Training"):
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Calculate average training loss across all training samples
            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            dice_metric.reset()
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                    outputs = self.model(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()

                    # Calculate Dice score
                    pred_label = torch.argmax(outputs, dim=1, keepdim=True)
                    num_classes = outputs.shape[1]
                    pred_onehot = F.one_hot(pred_label.squeeze(1).long(), num_classes=num_classes) \
                                    .permute(0,4,1,2,3).float()
                    label_onehot = F.one_hot(labels.squeeze(1).long(), num_classes=num_classes) \
                                    .permute(0,4,1,2,3).float()
                    dice_metric(y_pred=pred_onehot, y=label_onehot)
            
            # Calculate average validation loss and Dice score across all validation samples
            avg_val_loss = val_loss / len(val_loader)
            avg_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            epoch_time = time.time() - epoch_start

            # Log metrics
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_dice, epoch_time])

            # Print metrics
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Dice Score: {avg_dice:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s")

            # Save best model
            if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save(self.model.state_dict(), os.path.join(self.results_dir, "best_model.pth"))
                print(f"New best model saved! (Dice: {best_dice:.4f})")

        # Report training time
        total_time = time.time() - start_time
        print("\nTraining finished!")
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Best validation Dice score: {best_dice:.4f}")
        print(f"Results saved in: {self.results_dir}")

    def test(self, test_path: str):
        # Create test dataset
        test_files = glob(os.path.join(test_path, "*.pt"))
        test_ds = TorchDataset(test_files, transform=None)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

        # Load best model
        self.model.load_state_dict(torch.load(os.path.join(self.results_dir, "best_model.pth")))
        self.model.eval()

        # Initialize metrics
        loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        test_loss = 0.0
        dice_metric.reset()

        # Test loop
        print("\nRunning inference on test data...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                images, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()

                # Calculate Dice score
                pred_label = torch.argmax(outputs, dim=1, keepdim=True)
                num_classes = outputs.shape[1]
                pred_onehot = F.one_hot(pred_label.squeeze(1).long(), num_classes=num_classes) \
                                .permute(0,4,1,2,3).float()
                label_onehot = F.one_hot(labels.squeeze(1).long(), num_classes=num_classes) \
                                .permute(0,4,1,2,3).float()
                dice_metric(y_pred=pred_onehot, y=label_onehot)

                # Save first test prediction as PNG
                if i == 1:
                    import matplotlib.pyplot as plt
                    
                    # Take middle slice of first test sample
                    slice_idx = pred_label.shape[3] // 2
                    pred_slice = pred_label[0, 0, :, :, slice_idx].cpu()
                    label_slice = labels[0, 0, :, :, slice_idx].cpu()
                    image_slice = images[0, :, :, :, slice_idx].cpu()  # all modalities
                    
                    # Create overlay plot for each modality
                    modalities = ['FLAIR', 'T1', 'T1CE', 'T2']
                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                    
                    for i, modality in enumerate(modalities):
                        # Plot prediction overlay
                        axes[i, 0].imshow(image_slice[i], cmap='gray')
                        axes[i, 0].imshow(pred_slice, alpha=0.3)
                        axes[i, 0].set_title(f'{modality} with Prediction')
                        
                        # Plot ground truth overlay
                        axes[i, 1].imshow(image_slice[i], cmap='gray')
                        axes[i, 1].imshow(label_slice, alpha=0.3)
                        axes[i, 1].set_title(f'{modality} with Ground Truth')
                        
                        # Remove axes
                        axes[i, 0].axis('off')
                        axes[i, 1].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.results_dir, 'test_prediction_overlay.png'))
                    plt.close()

        # Calculate average test loss and dice score
        avg_test_loss = test_loss / len(test_loader)
        avg_dice = dice_metric.aggregate().item()

        # Save test results
        with open(os.path.join(self.results_dir, 'test_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Test Loss', 'Test Dice Score'])
            writer.writerow([avg_test_loss, avg_dice])

        print(f"\nTest Results:")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Dice Score: {avg_dice:.4f}")
        print(f"Results saved in: {self.results_dir}")

if __name__ == "__main__":
    train_path = "data\\split_dataset\\train"
    val_path = "data\\split_dataset\\val"
    results_path = "models\\unet\\results"
    test_path = "data\\split_dataset\\test"
    epochs = 200

    unet = UnetModel(train_path, val_path, results_path)
    unet.train(epochs)
    unet.test(test_path) 