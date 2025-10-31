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

    def profile_performance(self, train_loader, optimizer, loss_fn):
        """Profile the training performance for a few steps with comprehensive metrics."""
        self.model.train()
        scaler = GradScaler()
        
        # Create profiler output directory
        profile_dir = os.path.join(self.results_dir, "profiler_output")
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

    def train(self, profile: bool = False):
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

        if profile:
            # Run profiler for performance analysis
            print("\nRunning performance profiling...")
            self.profile_performance(train_loader, optimizer, loss_fn)
            print("\nProfiling completed\n")
            return
        
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
            torch.save(checkpoint, os.path.join(self.results_dir, "latest_checkpoint.pth"))

            # Save best model
            if avg_dice > best_dice:
                best_dice = avg_dice
                # Save both state dict and full checkpoint for best model
                torch.save(self.model.state_dict(), os.path.join(self.results_dir, "best_model.pth"))
                torch.save(checkpoint, os.path.join(self.results_dir, f"best_checkpoint_epoch_{epoch+1}.pth"))
                print(f"New best model saved! (Dice: {best_dice:.4f})")

        # Report training time
        total_time = time.time() - start_time
        print("\nTraining finished!")
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Best validation Dice score: {best_dice:.4f}")
        print(f"Results saved in: {self.results_dir}")

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

    #Runs sliding window inference (0.7 overlap) + TTA flips (adds 1–3% Dice often). Didn't improve DICE however.
    def test_sliding_window(self, best_model_path: str):
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

        print("\nRunning inference on test data...")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                # Move data to device
                images = batch["image"].to(self.device)  # shape: [1, 4, 128, 128, 128]
                labels = batch["label"].to(self.device)  # shape: [1, 1, 128, 128, 128]
                
                # Run sliding window inference with TTA
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                
                def tta_inference(x):
                    # Original prediction
                    pred = sliding_window_inference(
                        x, roi_size, sw_batch_size, 
                        self.model,
                        overlap=0.7,
                        mode='gaussian'
                    )
                    
                    # Flip H
                    pred_h = sliding_window_inference(
                        torch.flip(x, dims=(-3,)), roi_size, sw_batch_size,
                        self.model,
                        overlap=0.7,
                        mode='gaussian'
                    )
                    pred_h = torch.flip(pred_h, dims=(-3,))
                    
                    # Flip W
                    pred_w = sliding_window_inference(
                        torch.flip(x, dims=(-2,)), roi_size, sw_batch_size,
                        self.model,
                        overlap=0.7,
                        mode='gaussian'
                    )
                    pred_w = torch.flip(pred_w, dims=(-2,))
                    
                    # Flip D
                    pred_d = sliding_window_inference(
                        torch.flip(x, dims=(-1,)), roi_size, sw_batch_size,
                        self.model,
                        overlap=0.7,
                        mode='gaussian'
                    )
                    pred_d = torch.flip(pred_d, dims=(-1,))
                    
                    # Average all predictions
                    return torch.mean(torch.stack([pred, pred_h, pred_w, pred_d]), dim=0)
                
                # Run inference with TTA
                outputs = tta_inference(images)
                
                # Calculate loss
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
        epochs=1)
    
    swinunetr.train()
    swinunetr.test(best_model_path="models\\swinunetr\\results\\96i_24f_results")
