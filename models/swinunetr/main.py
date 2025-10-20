import os
import time
import csv
import torch
import torch.nn.functional as F
from glob import glob
from torch.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd
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
from monai.optimizers import WarmupCosineSchedule

# Define custom dataset that loads torch tensors 
class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        sample = {"image": data["image"], "label": data["label"]}

        # Apply transforms if any
        if self.transform:
            sample = self.transform(sample)

        return sample
    
class SwinUnetrModel:
    def __init__(self, 
                 image_size, 
                 feature_size, 
                 learning_rate, 
                 num_workers,
                 prefetch_factor,
                 gradient_accumulation_steps, 
                 gradient_checkpointing, 
                 cudnn_checkpointing, 
                 epochs):
        
        self.train_path = "data\\split_dataset\\train"
        self.val_path = "data\\split_dataset\\val"
        self.test_path = "data\\split_dataset\\test"
        self.results_path = "models\\swinunetr\\results"
        self.feature_size = feature_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs

        # Enable cuDNN benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = cudnn_checkpointing
        
        #  Get list of preprocessed training and validation .pt files 
        self.train_files = glob(os.path.join(self.train_path, "*.pt"))
        self.val_files = glob(os.path.join(self.val_path, "*.pt"))
        self.results_dir = self.results_path

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}") 

        # Define UNet model 
        self.model = SwinUNETR(
            img_size=(self.image_size, self.image_size, self.image_size), 
            in_channels=4,            
            out_channels=4,           
            feature_size=self.feature_size,          
            drop_rate=0.2,                          # Added dropout for regularization (since dataset is small)
            attn_drop_rate=0.2,                     # Added attention dropout (since dataset is small)
            dropout_path_rate=0.2,                  # Added drop path (since dataset is small)
            use_checkpoint=gradient_checkpointing   # Gradient checkpointing to reduce memory usage
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

        # Create data loaders with pinned memory for faster CPU->GPU transfer
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
        
        # Use ADAM with learning rate 1e-4
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
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

        # Run profiler for performance analysis
        # print("\nRunning performance profiling...")
        # self.profile_performance(train_loader, optimizer, loss_fn)
        # print("\nProfiling completed. Continuing with training...\n")

        best_dice = 0.0
        start_time = time.time()

        # Training loop (# of epochs)
        for epoch in range(self.epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            scaler = GradScaler()  # for mixed precision training
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                # Non-blocking transfer to GPU
                # While GPU is processing previous batch, CPU can immediately prepare next batch
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                                
                # Mixed precision training
                with autocast(device_type=self.device.type):
                    outputs = self.model(images)
                    loss = loss_fn(outputs, labels)
                    # Normalize loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            
                # Update weights only after accumulating gradients
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)  # slightly more efficient
                
                # Add synchronization point for accurate profiling
                #torch.cuda.synchronize()
                
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

    def profile_performance(self, train_loader, optimizer, loss_fn, num_steps=100):
        """Profile the training performance for a few steps."""
        self.model.train()
        scaler = GradScaler()
        
        # Configure profiler
        activities = [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ]
        
        with profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=5,        # Number of steps to wait before logging
                warmup=5,      # Number of steps for warmup
                active=20,     # Number of steps to log
                repeat=1       # Repeat the schedule these many times
            ),
            record_shapes=True,
            with_stack=True
        ) as prof:
            
            for step, batch in enumerate(train_loader):
                if step >= 30:  # 5 (wait) + 5 (warmup) + 20 (active)
                    break
                    
                with record_function("train_step"):
                    # Get the data
                    with record_function("data_transfer"):
                        images = batch["image"].to(self.device)
                        labels = batch["label"].to(self.device)
                    
                    # Forward pass
                    with record_function("forward"), autocast(device_type=self.device.type):
                        outputs = self.model(images)
                        loss = loss_fn(outputs, labels)
                    
                    # Backward pass
                    with record_function("backward"):
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                
                prof.step()
        
        # Print profiler summary
        print("\nProfiler Results Summary:")
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=50))
        #prof.export_chrome_trace("trace.json")

    def test(self, test_path):
        # Run sliding window inference (0.7 overlap) + TTA flips (adds 1–3% Dice often)
        pass

if __name__ == "__main__":
    swinunetr = SwinUnetrModel(
        image_size=96,
        feature_size=24,
        learning_rate=1e-4,
        num_workers=8,                    # Number of parallel DataLoader workers
        prefetch_factor=None,             # Number of batches to prefetch per worker in DataLoader
        gradient_accumulation_steps=4,    # Emulate batch size of 4
        gradient_checkpointing = True,
        cudnn_checkpointing=True,
        epochs=100)
    
    swinunetr.train()
