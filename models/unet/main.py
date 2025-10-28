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
import matplotlib.pyplot as plt
from monai.optimizers import WarmupCosineSchedule

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
    def __init__(self, 
                 learning_rate, 
                 num_workers,
                 prefetch_factor,
                 gradient_accumulation_steps,
                 cudnn_checkpointing,
                 epochs):
        
        self.train_path = "data\\split_dataset\\train"
        self.val_path = "data\\split_dataset\\val"
        self.test_path = "data\\split_dataset\\test"
        self.results_path = "models\\unet\\results"
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs

        #Enable cuDNN benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = cudnn_checkpointing

        #  Get list of preprocessed training and validation .pt files 
        self.train_files = glob(os.path.join(self.train_path, "*.pt"))
        self.val_files = glob(os.path.join(self.val_path, "*.pt"))
        self.results_dir = self.results_path

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

    def train(self):
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
        train_loader = DataLoader(
            train_ds, 
            batch_size=1, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True
        )
        
        # Use ADAM with learning rate 1e-4 and fused implementation
        if hasattr(torch.optim.Adam, 'fused') and torch.cuda.is_available():
            # Use fused Adam implementation if available (much faster)
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=1e-5,
                fused=True  # Enable fused implementation
            )
        else:
            # Fall back to normal Adam if fused not available
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=1e-5
            )

        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=50,  # Warmup for first 50 steps 
            t_total=self.epochs * len(train_loader) // self.gradient_accumulation_steps,  # Total optimization steps
        )
        
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
        for epoch in range(self.epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            optimizer.zero_grad(set_to_none=True)  # Zero gradients at start
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                # Non-blocking transfer to GPU
                # While GPU is processing previous batch, CPU can immediately prepare next batch
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = loss_fn(outputs, labels) / self.gradient_accumulation_steps
                loss.backward()

                # Update weights only after accumulating gradients
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)  # slightly more efficient
                
                train_loss += loss.item()

            # Calculate average training loss across all training samples
            avg_train_loss = train_loss / len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            dice_metric.reset()
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    images = batch["image"].to(self.device, non_blocking=True)
                    labels = batch["label"].to(self.device, non_blocking=True)
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

            # Clear GPU cache after validation
            torch.cuda.empty_cache()

            # Log metrics
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_dice, epoch_time])

            # Print metrics
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Dice Score: {avg_dice:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s")

            # Save full checkpoint after each epoch (overwriting previous checkpoint)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'dice_score': avg_dice,
                'best_dice': best_dice
            }
            torch.save(checkpoint, os.path.join(self.results_dir, "latest_checkpoint.pth"))

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

    def test(self):
        # Create test dataset
        test_files = glob(os.path.join(self.test_path, "*.pt"))
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
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
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

                # Save a test prediction as PNG
                if i == 1:                    
                    # Create results directory if it doesn't exist
                    os.makedirs(os.path.join(self.results_dir, "\\images"), exist_ok=True)

                    # Take middle slice of first test sample
                    slice_idx = pred_label.shape[3] // 2
                    pred_slice = pred_label[0, 0, :, :, slice_idx].cpu()
                    label_slice = labels[0, 0, :, :, slice_idx].cpu()
                    image_slice = images[0, :, :, :, slice_idx].cpu()  # all modalities
                    
                    # Create overlay plot for each modality
                    modalities = ['FLAIR', 'T1', 'T1CE', 'T2']
                    fig, axes = plt.subplots(4, 2, figsize=(10, 20))
                    
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
    # New UNET model is running with the following to match SwinUNETR: 
    # cudnn_checkpointing enabled, 
    # gradient accumulation steps = 4,
    # DataLoader: prefetch_factor=None, num_workers=8, pin_memory=True, persistent_workers=True
    # Non-blocking transfers to GPU
    # Adam optimizer with fused implementation
    # Clear GPU cache after validation
    # Learning rate scheduler: WarmupCosineSchedule
    # Save checkpoint after every epoch
    unet = UnetModel(
        learning_rate=1e-4,
        num_workers=8,
        prefetch_factor=None,
        gradient_accumulation_steps=4,
        cudnn_checkpointing=True,
        epochs=200)
    
    unet.train()
    unet.test() 