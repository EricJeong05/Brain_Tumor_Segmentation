import os
import time
import csv
import torch
import torch.nn.functional as F
from glob import glob
from monai.data import DataLoader
from tqdm import tqdm
from monai.transforms import (
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    Compose
)
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
import numpy as np
import matplotlib.pyplot as plt

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
    
class SwinUnetrModel:
    def __init__(self, train_path: str, val_path: str, results_path: str):
        #  Get list of preprocessed training and validation .pt files 
        self.train_files = glob(os.path.join(train_path, "*.pt"))
        self.val_files = glob(os.path.join(val_path, "*.pt"))
        self.results_dir = results_path

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}") 

        # Define UNet model 
        self.model = SwinUNETR(
            img_size=(128, 128, 128), # BraTS standard size
            in_channels=4,            # Keep if using BraTS
            out_channels=4,           # Keep if using BraTS
            feature_size=48,          # Paper default
            drop_rate=0.2,            # Added dropout for regularization
            attn_drop_rate=0.2,       # Added attention dropout
            dropout_path_rate=0.2,    # Added drop path
            use_checkpoint=True,      # Keep for memory efficiency
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

        # Create data loaders with pinned memory for faster GPU transfer
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
            
            # Calculate average validation loss and Dice score
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

if __name__ == "__main__":
    train_path = "data\\split_dataset\\train"
    val_path = "data\\split_dataset\\val"
    test_path = "data\\split_dataset\\test"
    results_path = "models\\swinunetr\\results"
    epochs = 5

    swinunetr = SwinUnetrModel(train_path, val_path, results_path)
    swinunetr.train(epochs)
    #swinunetr.test(test_path)