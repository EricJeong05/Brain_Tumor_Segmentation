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
import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.custom_swin_unetr.custom_swin_unetr import CustomSwinUNETR

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.optimizers import WarmupCosineSchedule
import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference

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

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}") 

        # Define UNet model 
        self.model = CustomSwinUNETR(
            img_size=(self.image_size, self.image_size, self.image_size), 
            in_channels=4,            
            out_channels=4,           
            feature_size=self.feature_size,          
            drop_rate=0.2,                          # Added dropout for regularization (since dataset is small)
            attn_drop_rate=0.2,                     # Added attention dropout (since dataset is small)
            dropout_path_rate=0.2,                  # Added drop path (since dataset is small)
            use_checkpoint=gradient_checkpointing   # Gradient checkpointing to reduce memory usage
        ).to(self.device)
        
    def profile_performance(self, results_path, train_loader, optimizer, loss_fn):
        """Profile the training performance for a few steps with comprehensive metrics."""
        self.model.train()
        scaler = GradScaler()
        
        # Create profiler output directory
        profile_dir = os.path.join(results_path, "profiler_output")
        os.makedirs(profile_dir, exist_ok=True)
        
        # Configure profiler with expanded activities
        activities = [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA
        ]
        
        with profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=2,        # Number of steps to wait before logging
                warmup=3,      # Number of steps for warmup
                active=25,     # Number of steps to log
                repeat=1       # Repeat the schedule these many times
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True
        ) as prof:
            
            for step, batch in enumerate(train_loader):
                if step >= 30:  # 2 (wait) + 3 (warmup) + 25 (active)
                    break
                    
                with record_function("train_step"):
                    # Data loading and transfer
                    with record_function("data_transfer"):
                        images = batch["image"].to(self.device, non_blocking=True)
                        labels = batch["label"].to(self.device, non_blocking=True)
                    
                    # Forward pass with mixed precision
                    with record_function("forward"), autocast(device_type=self.device.type):
                        with record_function("model_forward"):
                            outputs = self.model(images)
                        with record_function("loss_computation"):
                            loss = loss_fn(outputs, labels)
                    
                    # Backward pass
                    with record_function("backward"):
                        optimizer.zero_grad(set_to_none=True)
                        with record_function("backward_loss"):
                            scaler.scale(loss).backward()
                        with record_function("optimizer_step"):
                            scaler.step(optimizer)
                            scaler.update()
                
                prof.step()
        
        # Generate comprehensive profiling report
        profile_report = os.path.join(profile_dir, "profiling_report.txt")
        with open(profile_report, 'w') as f:
            # Overall profiling summary
            f.write("=== PyTorch Profiler Report ===\n\n")
            f.write("1. CUDA Kernel Analysis (Top 20 by CUDA time):\n")
            f.write(prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=20,
                top_level_events_only=False
            ))
            
            f.write("\n\n2. CUDA Memory Usage Analysis (Top 20 by CUDA Mem):\n")
            f.write(prof.key_averages().table(
                sort_by="self_cuda_memory_usage",
                row_limit=20
            ))
            
            f.write("\n\n3. CPU Operation Analysis (Top 20 by CPU time):\n")
            f.write(prof.key_averages().table(
                sort_by="cpu_time_total",
                row_limit=20
            ))
            
            # Detailed analysis of key operations
            f.write("\n\n4. Detailed Operation Analysis:\n")            
            key_ops = ["data_transfer", "model_forward", "loss_computation", 
                      "backward_loss", "optimizer_step"]
            
            for event_name in key_ops:
                events = [e for e in prof.events() if event_name in e.name]
                if events:
                    avg_cuda_time = sum(e.device_time for e in events) / len(events)
                    avg_cpu_time = sum(e.cpu_time for e in events) / len(events)
                    avg_memory = sum(e.device_memory_usage for e in events if e.device_memory_usage is not None) / len(events)
                    
                    f.write(f"\nOperation: {event_name}\n")
                    f.write(f"Average CUDA Time: {avg_cuda_time/1000:.2f}ms\n")
                    f.write(f"Average CPU Time: {avg_cpu_time/1000:.2f}ms\n")
                    f.write(f"Average CUDA Memory Usage: {avg_memory/1024/1024:.2f}MB\n")
        
        print(f"\nProfiling results saved to: {profile_report}")
        prof.export_chrome_trace(os.path.join(profile_dir, "trace.json"))

    def resume_from_checkpoint(self, checkpoint_path: str, results_path: str = "models\\swinunetr\\results"):
        """Resume training from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to the folder containing latest_checkpoint.pth
            results_path: Path to save continued training results (will create new CSV)
        """
        # Load checkpoint
        checkpoint_file = os.path.join(checkpoint_path, "latest_checkpoint.pth")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        print(f"Loading checkpoint from: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        
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
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        
        # Use AdamW with learning rate 1e-4 and fused implementation
        if hasattr(torch.optim.AdamW, 'fused') and torch.cuda.is_available():
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=1e-2,
                fused=True
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=1e-2
            )
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded optimizer state")
            
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=50,
            t_total=self.epochs * len(train_loader) // self.gradient_accumulation_steps,
        )
        
        # Load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded scheduler state")

        # Use Dice + CrossEntropy loss
        loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        # Use Dice metric for evaluation
        dice_metric = DiceMetric(include_background=False, reduction="mean")

        # Create results directory if it doesn't exist
        os.makedirs(results_path, exist_ok=True)
        
        # Initialize CSV logging with new filename
        csv_filename = os.path.join(results_path, f"training_results_from_checkpoint.csv")
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Dice Score', 'Time (s)'])

        # Resume from checkpoint epoch
        start_epoch = checkpoint['epoch']
        best_dice = checkpoint.get('best_dice', 0.0)
        print(f"\nResuming training from epoch {start_epoch + 1}")
        print(f"Best dice so far: {best_dice:.4f}")
        print(f"Training until epoch {self.epochs}\n")
        
        start_time = time.time()

        # Training loop (continue from checkpoint epoch)
        for epoch in range(start_epoch, self.epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            scaler = GradScaler()
            optimizer.zero_grad(set_to_none=True)

            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                                
                with autocast(device_type=self.device.type):
                    outputs = self.model(images)
                    loss = loss_fn(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                                
                train_loss += loss.item()

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

                    pred_label = torch.argmax(outputs, dim=1, keepdim=True)
                    num_classes = outputs.shape[1]
                    pred_onehot = F.one_hot(pred_label.squeeze(1).long(), num_classes=num_classes) \
                                    .permute(0,4,1,2,3).float()
                    label_onehot = F.one_hot(labels.squeeze(1).long(), num_classes=num_classes) \
                                    .permute(0,4,1,2,3).float()
                    dice_metric(y_pred=pred_onehot, y=label_onehot)
            
            avg_val_loss = val_loss / len(val_loader)
            avg_dice = dice_metric.aggregate().item()
            dice_metric.reset()
            epoch_time = time.time() - epoch_start

            torch.cuda.empty_cache()

            # Log metrics
            with open(csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_dice, epoch_time])

            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {avg_val_loss:.4f}")
            print(f"Dice Score: {avg_dice:.4f}")
            print(f"Epoch Time: {epoch_time:.2f}s")

            # Save full checkpoint after each epoch
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
            torch.save(checkpoint, os.path.join(results_path, "latest_checkpoint.pth"))

            # Save best model
            if avg_dice > best_dice:
                best_dice = avg_dice
                torch.save(self.model.state_dict(), os.path.join(results_path, "best_model.pth"))
                torch.save(checkpoint, os.path.join(results_path, f"best_model_checkpoint.pth"))
                print(f"New best model saved! (Dice: {best_dice:.4f})")

        # Report training time
        total_time = time.time() - start_time
        print("\nTraining finished!")
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Best validation Dice score: {best_dice:.4f}")
        print(f"Results saved in: {results_path}")

    def train(self, results_path: str = "models\\swinunetr\\results", profile: bool = False):
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
        
        # Use AdamW with learning rate 1e-4 and fused implementation
        if hasattr(torch.optim.AdamW, 'fused') and torch.cuda.is_available():
            # Use fused AdamW implementation if available (much faster)
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=1e-2,  # Higher weight decay for AdamW (0.01 is typical)
                fused=True  # Enable fused implementation
            )
        else:
            # Fall back to normal AdamW if fused not available
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=1e-2
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
        os.makedirs(results_path, exist_ok=True)

        if profile:
            # Run profiler for performance analysis
            print("\nRunning performance profiling...")
            self.profile_performance(results_path, train_loader, optimizer, loss_fn)
            print("\nProfiling completed\n")
            return
        
        # Initialize CSV logging
        csv_filename = os.path.join(results_path, f"training_results.csv")
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
            scaler = GradScaler()  # for mixed precision training
            optimizer.zero_grad(set_to_none=True)  # Zero gradients at start

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
            torch.save(checkpoint, os.path.join(results_path, "latest_checkpoint.pth"))

            # Save best model
            if avg_dice > best_dice:
                best_dice = avg_dice
                # Save both state dict and full checkpoint for best model
                torch.save(self.model.state_dict(), os.path.join(results_path, "best_model.pth"))
                torch.save(checkpoint, os.path.join(results_path, f"best_model_checkpoint.pth"))
                print(f"New best model saved! (Dice: {best_dice:.4f})")

        # Report training time
        total_time = time.time() - start_time
        print("\nTraining finished!")
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Best validation Dice score: {best_dice:.4f}")
        print(f"Results saved in: {results_path}")

    def test(self, best_model_path: str):
        # Create test dataset
        test_files = glob(os.path.join(self.test_path, "*.pt"))
        test_ds = TorchDataset(test_files, transform=None)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

        # Load best model
        self.model.load_state_dict(torch.load(os.path.join(best_model_path, "best_model.pth")))
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
                    os.makedirs(os.path.join(best_model_path, "\\images"), exist_ok=True)

                    # Take middle slice of first test sample
                    slice_idx = pred_label.shape[3] // 2
                    pred_slice = pred_label[0, 0, :, :, slice_idx].cpu()
                    label_slice = labels[0, 0, :, :, slice_idx].cpu()
                    image_slice = images[0, :, :, :, slice_idx].cpu()  # all modalities
                    
                    # Create overlay plot for each modality
                    # Rows: Prediction, Ground Truth
                    # Columns: FLAIR, T1, T1CE, T2
                    modalities = ['FLAIR', 'T1', 'T1CE', 'T2']
                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
                    
                    for j, modality in enumerate(modalities):
                        # Plot prediction overlay (row 0)
                        axes[0, j].imshow(image_slice[j], cmap='gray')
                        axes[0, j].imshow(pred_slice, alpha=0.3)
                        axes[0, j].set_title(f'{modality} - Prediction')
                        axes[0, j].axis('off')
                        
                        # Plot ground truth overlay (row 1)
                        axes[1, j].imshow(image_slice[j], cmap='gray')
                        axes[1, j].imshow(label_slice, alpha=0.3)
                        axes[1, j].set_title(f'{modality} - Ground Truth')
                        axes[1, j].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(best_model_path, 'test_prediction_overlay.png'))
                    plt.close()

        # Calculate average test loss and dice score
        avg_test_loss = test_loss / len(test_loader)
        avg_dice = dice_metric.aggregate().item()

        # Save test results
        with open(os.path.join(best_model_path, 'test_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Test Loss', 'Test Dice Score'])
            writer.writerow([avg_test_loss, avg_dice])

        print(f"\nTest Results:")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Dice Score: {avg_dice:.4f}")
        print(f"Results saved in: {best_model_path}")

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
    
    #swinunetr.train(
    #    results_path = "models\\swinunetr\\results\\customswinunetr_results\\first_pass", 
    #    profile=False)

    swinunetr.resume_from_checkpoint(
        checkpoint_path="models\\swinunetr\\results\\customswinunetr_results\\first_pass",
        results_path="models\\swinunetr\\results\\customswinunetr_results\\first_pass")
    
    swinunetr.test(best_model_path="models\\swinunetr\\results\\96i_24f_results")