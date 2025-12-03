import os
import time
import csv
import torch
import torch.nn.functional as F
from glob import glob
from torch.amp import autocast, GradScaler
import numpy as np
from sklearn.model_selection import KFold
from monai.data import DataLoader
from tqdm import tqdm
from monai.transforms import (
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    Compose
)
import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.custom_swin_unetr.custom_swin_unetr import CustomSwinUNETR

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.optimizers import WarmupCosineSchedule
import matplotlib.pyplot as plt

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
    
class SwinUnetrCrossValidation:
    def __init__(self, 
                 image_size, 
                 feature_size, 
                 learning_rate, 
                 num_workers,
                 prefetch_factor,
                 gradient_accumulation_steps, 
                 gradient_checkpointing, 
                 cudnn_checkpointing, 
                 epochs,
                 n_folds,
                 early_stopping_patience):
        
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
        self.n_folds = n_folds
        self.early_stopping_patience = early_stopping_patience
        self.gradient_checkpointing = gradient_checkpointing

        # Enable cuDNN benchmarking for consistent input sizes
        torch.backends.cudnn.benchmark = cudnn_checkpointing
        
        # Combine train and val for k-fold CV (they'll be split in train_cross_validation)
        train_files = glob(os.path.join(self.train_path, "*.pt"))
        val_files = glob(os.path.join(self.val_path, "*.pt"))
        self.all_train_files = train_files + val_files
        self.test_files = glob(os.path.join(self.test_path, "*.pt"))

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}") 
        print(f"Total training samples: {len(self.all_train_files)}")
        print(f"Test samples: {len(self.test_files)}")
        
        self.model = None  # Will be created per fold
    
    def create_model(self):
        """Create a fresh model for each fold."""
        return CustomSwinUNETR(
            img_size=(self.image_size, self.image_size, self.image_size), 
            in_channels=4,            
            out_channels=4,           
            feature_size=self.feature_size,          
            drop_rate=0.2,
            attn_drop_rate=0.2,
            dropout_path_rate=0.2,
            use_checkpoint=self.gradient_checkpointing
        ).to(self.device)

    def train_cross_validation(self, results_path: str = "models\\swinunetr\\results"):
        """Train 5-fold cross-validation with early stopping."""
        print(f"\n{'='*80}")
        print(f"Starting {self.n_folds}-Fold Cross-Validation Training")
        print(f"Total samples: {len(self.all_train_files)}")
        print(f"Epochs per fold: {self.epochs}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"{'='*80}\n")
        
        # Create K-Fold splits (90/10 train/val split)
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # Store fold results
        fold_results = []
        fold_models = []
        
        cv_start_time = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.all_train_files)):
            # Only run for n_folds iterations (5 folds with 90/10 split each)
            if fold >= self.n_folds:
                break

            print(f"\n{'='*80}")
            print(f"FOLD {fold + 1}/{self.n_folds}")
            print(f"{'='*80}")
            
            # Create fold-specific results directory
            fold_results_path = os.path.join(results_path, f"fold_{fold + 1}")
            os.makedirs(fold_results_path, exist_ok=True)
            
            # Split data for this fold
            train_files = [self.all_train_files[i] for i in train_idx]
            val_files = [self.all_train_files[i] for i in val_idx]
            
            print(f"Train samples: {len(train_files)}")
            print(f"Validation samples: {len(val_files)}\n")
            
            # Train single fold
            best_dice, best_model_path = self.train_single_fold(
                fold + 1, 
                train_files, 
                val_files, 
                fold_results_path
            )
            
            fold_results.append({
                'fold': fold + 1,
                'best_dice': best_dice,
                'model_path': best_model_path
            })
            fold_models.append(best_model_path)
            
            print(f"\nFold {fold + 1} completed. Best Dice: {best_dice:.4f}")
            print(f"Model saved: {best_model_path}")
        
        # Calculate cross-validation statistics
        cv_end_time = time.time()
        total_cv_time = (cv_end_time - cv_start_time) / 3600  # hours
        
        dice_scores = [r['best_dice'] for r in fold_results]
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {total_cv_time:.2f} hours")
        print(f"\nFold Results:")
        for r in fold_results:
            print(f"  Fold {r['fold']}: Dice = {r['best_dice']:.4f}")
        print(f"\nMean Dice: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"Best Fold: {max(fold_results, key=lambda x: x['best_dice'])['fold']} (Dice: {max(dice_scores):.4f})")
        print(f"{'='*80}\n")
        
        # Save cross-validation summary
        summary_path = os.path.join(results_path, 'cv_summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Fold', 'Best Dice', 'Model Path'])
            for r in fold_results:
                writer.writerow([r['fold'], r['best_dice'], r['model_path']])
            writer.writerow(['Mean', mean_dice, ''])
            writer.writerow(['Std', std_dice, ''])
        
        print(f"Cross-validation summary saved: {summary_path}")
        
        return fold_models, fold_results
    
    def resume_fold_from_checkpoint(self, fold_num, checkpoint_path, results_path):
        """Resume training a single fold from checkpoint.
        
        Args:
            fold_num: The fold number (1-5)
            checkpoint_path: Path to the folder containing latest_checkpoint.pth
            results_path: Path to save continued training results
        """
        # Load checkpoint
        checkpoint_file = os.path.join(checkpoint_path, "latest_checkpoint.pth")
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        print(f"\nLoading checkpoint for fold {fold_num} from: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        
        # Verify fold number matches
        if checkpoint.get('fold') != fold_num:
            print(f"Warning: Checkpoint fold ({checkpoint.get('fold')}) doesn't match requested fold ({fold_num})")
        
        # Create model and load state
        self.model = self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        
        # Note: We need the original train/val split - this should be provided
        # For now, we'll reconstruct the split using the same random seed
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        splits = list(kfold.split(self.all_train_files))
        train_idx, val_idx = splits[fold_num - 1]
        train_files = [self.all_train_files[i] for i in train_idx]
        val_files = [self.all_train_files[i] for i in val_idx]
        
        # Real-time augmentations
        train_transforms = Compose([
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1)
        ])
        
        train_ds = TorchDataset(train_files, transform=train_transforms)
        val_ds = TorchDataset(val_files, transform=None)
        
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True,
            prefetch_factor=self.prefetch_factor, persistent_workers=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            prefetch_factor=self.prefetch_factor, persistent_workers=True
        )
        
        # Optimizer with fused AdamW
        if hasattr(torch.optim.AdamW, 'fused') and torch.cuda.is_available():
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate,
                weight_decay=1e-2, fused=True
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2
            )
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded optimizer state")
        
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=50,
            t_total=self.epochs * len(train_loader) // self.gradient_accumulation_steps
        )
        
        # Load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded scheduler state")
        
        loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Initialize CSV logging with new filename
        csv_filename = os.path.join(results_path, "training_results_from_checkpoint.csv")
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Dice Score', 'Time (s)'])
        
        # Resume from checkpoint
        start_epoch = checkpoint['epoch']
        best_dice = checkpoint.get('best_dice', 0.0)
        best_epoch = start_epoch
        patience_counter = 0
        best_model_path = os.path.join(results_path, "best_model.pth")
        
        print(f"Resuming training from epoch {start_epoch + 1}")
        print(f"Best dice so far: {best_dice:.4f}")
        print(f"Training until epoch {self.epochs}\n")
        
        fold_start_time = time.time()
        
        # Training loop (continue from checkpoint epoch)
        for epoch in range(start_epoch, self.epochs):
            epoch_start = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            scaler = GradScaler()
            optimizer.zero_grad(set_to_none=True)
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")):
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
                for batch in val_loader:
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
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Dice: {avg_dice:.4f}, Time: {epoch_time:.1f}s")
            
            # Save full checkpoint after each epoch
            checkpoint = {
                'epoch': epoch + 1,
                'fold': fold_num,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'dice_score': avg_dice,
                'best_dice': best_dice
            }
            torch.save(checkpoint, os.path.join(results_path, "latest_checkpoint.pth"))
            
            # Early stopping logic (only if patience > 0)
            if self.early_stopping_patience > 0:
                if avg_dice > best_dice:
                    best_dice = avg_dice
                    best_epoch = epoch + 1
                    patience_counter = 0
                    # Save both state dict and full checkpoint for best model
                    torch.save(self.model.state_dict(), best_model_path)
                    best_checkpoint = checkpoint.copy()
                    best_checkpoint['best_dice'] = best_dice
                    torch.save(best_checkpoint, os.path.join(results_path, "best_model_checkpoint.pth"))
                    print(f"  ✓ New best model saved! (Dice: {best_dice:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{self.early_stopping_patience})")
                
                # Check early stopping
                if patience_counter >= self.early_stopping_patience:
                    print(f"\n  Early stopping triggered at epoch {epoch + 1}")
                    print(f"  Best dice {best_dice:.4f} at epoch {best_epoch}")
                    break
            else:
                # No early stopping - just track best model
                if avg_dice > best_dice:
                    best_dice = avg_dice
                    best_epoch = epoch + 1
                    # Save both state dict and full checkpoint for best model
                    torch.save(self.model.state_dict(), best_model_path)
                    best_checkpoint = checkpoint.copy()
                    best_checkpoint['best_dice'] = best_dice
                    torch.save(best_checkpoint, os.path.join(results_path, "best_model_checkpoint.pth"))
                    print(f"  ✓ New best model saved! (Dice: {best_dice:.4f})")
        
        fold_time = (time.time() - fold_start_time) / 60
        print(f"\nFold training time: {fold_time:.1f} minutes")
        print(f"Best validation Dice: {best_dice:.4f} (epoch {best_epoch})")
        
        return best_dice, best_model_path

    def train_single_fold(self, fold_num, train_files, val_files, results_path):
        """Train a single fold with early stopping."""
        # Create fresh model for this fold
        self.model = self.create_model()
        
        # Real-time augmentations
        train_transforms = Compose([
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.1)
        ])
        
        train_ds = TorchDataset(train_files, transform=train_transforms)
        val_ds = TorchDataset(val_files, transform=None)
        
        train_loader = DataLoader(
            train_ds, batch_size=1, shuffle=True, 
            num_workers=self.num_workers, pin_memory=True,
            prefetch_factor=self.prefetch_factor, persistent_workers=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            prefetch_factor=self.prefetch_factor, persistent_workers=True
        )
        
        # Optimizer with fused AdamW
        if hasattr(torch.optim.AdamW, 'fused') and torch.cuda.is_available():
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate,
                weight_decay=1e-2, fused=True
            )
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.learning_rate, weight_decay=1e-2
            )
        
        # Reduced warmup steps for 50 epochs
        scheduler = WarmupCosineSchedule(
            optimizer=optimizer,
            warmup_steps=50,  
            t_total=self.epochs * len(train_loader) // self.gradient_accumulation_steps
        )
        
        loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Initialize CSV logging
        csv_filename = os.path.join(results_path, "training_results.csv")
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Train Loss', 'Val Loss', 'Dice Score', 'Time (s)'])
        
        # Early stopping variables
        best_dice = 0.0
        best_epoch = 0
        patience_counter = 0
        best_model_path = os.path.join(results_path, "best_model.pth")
        
        fold_start_time = time.time()
        
        # Training loop
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            scaler = GradScaler()
            optimizer.zero_grad(set_to_none=True)
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")):
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
                for batch in val_loader:
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
            
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Dice: {avg_dice:.4f}, Time: {epoch_time:.1f}s")
            
            # Save full checkpoint after each epoch (overwriting previous checkpoint)
            checkpoint = {
                'epoch': epoch + 1,
                'fold': fold_num,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'dice_score': avg_dice,
                'best_dice': best_dice
            }
            torch.save(checkpoint, os.path.join(results_path, "latest_checkpoint.pth"))
            # Early stopping logic (only if patience > 0)
            if self.early_stopping_patience > 0:
                if avg_dice > best_dice:
                    best_dice = avg_dice
                    best_epoch = epoch + 1
                    patience_counter = 0
                    # Save both state dict and full checkpoint for best model
                    torch.save(self.model.state_dict(), best_model_path)
                    best_checkpoint = checkpoint.copy()
                    best_checkpoint['best_dice'] = best_dice
                    torch.save(best_checkpoint, os.path.join(results_path, "best_model_checkpoint.pth"))
                    print(f"  ✓ New best model saved! (Dice: {best_dice:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{self.early_stopping_patience})")
                
                # Check early stopping
                if patience_counter >= self.early_stopping_patience:
                    print(f"\n  Early stopping triggered at epoch {epoch + 1}")
                    print(f"  Best dice {best_dice:.4f} at epoch {best_epoch}")
                    break
            else:
                # No early stopping - just track best model
                if avg_dice > best_dice:
                    best_dice = avg_dice
                    best_epoch = epoch + 1
                    # Save both state dict and full checkpoint for best model
                    torch.save(self.model.state_dict(), best_model_path)
                    best_checkpoint = checkpoint.copy()
                    best_checkpoint['best_dice'] = best_dice
                    torch.save(best_checkpoint, os.path.join(results_path, "best_model_checkpoint.pth"))
                    print(f"  ✓ New best model saved! (Dice: {best_dice:.4f})")
        
        fold_time = (time.time() - fold_start_time) / 60
        print(f"\nFold training time: {fold_time:.1f} minutes")
        print(f"Best validation Dice: {best_dice:.4f} (epoch {best_epoch})")
        
        return best_dice, best_model_path
    
    def ensemble_test(self, results_path: str):
        """Test using ensemble of all fold models.
        
        Args:
            results_path: Base path containing fold_1, fold_2, ..., fold_n directories
        """
        print(f"\n{'='*80}")
        print(f"ENSEMBLE TESTING WITH {self.n_folds} MODELS")
        print(f"{'='*80}\n")
        
        test_ds = TorchDataset(self.test_files, transform=None)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
        
        # Load all fold models from best_model.pth in each fold directory
        models = []
        for fold in range(1, self.n_folds + 1):
            model_path = os.path.join(results_path, f"fold_{fold}", "best_model.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            model = self.create_model()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            models.append(model)
            print(f"Loaded model from fold {fold}: {model_path}")
        
        loss_fn = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
        dice_metric = DiceMetric(include_background=False, reduction="mean")
        test_loss = 0.0
        dice_metric.reset()
        
        print("Running ensemble inference on test data...")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_loader)):
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                
                # Ensemble prediction: average outputs from all models
                ensemble_outputs = []
                for model in models:
                    outputs = model(images)
                    ensemble_outputs.append(outputs)
                
                # Average predictions
                ensemble_output = torch.stack(ensemble_outputs).mean(dim=0)
                
                loss = loss_fn(ensemble_output, labels)
                test_loss += loss.item()
                
                pred_label = torch.argmax(ensemble_output, dim=1, keepdim=True)
                num_classes = ensemble_output.shape[1]
                pred_onehot = F.one_hot(pred_label.squeeze(1).long(), num_classes=num_classes) \
                                .permute(0,4,1,2,3).float()
                label_onehot = F.one_hot(labels.squeeze(1).long(), num_classes=num_classes) \
                                .permute(0,4,1,2,3).float()
                dice_metric(y_pred=pred_onehot, y=label_onehot)
                
                # Save visualization for first test sample
                if i == 1:
                    os.makedirs(os.path.join(results_path, "images"), exist_ok=True)
                    slice_idx = pred_label.shape[3] // 2
                    pred_slice = pred_label[0, 0, :, :, slice_idx].cpu()
                    label_slice = labels[0, 0, :, :, slice_idx].cpu()
                    image_slice = images[0, :, :, :, slice_idx].cpu()
                    
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
                    plt.savefig(os.path.join(results_path, 'images', 'ensemble_prediction.png'))
                    plt.close()
        
        avg_test_loss = test_loss / len(test_loader)
        avg_dice = dice_metric.aggregate().item()
        
        # Save ensemble test results
        with open(os.path.join(results_path, 'ensemble_test_results.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Test Loss', avg_test_loss])
            writer.writerow(['Test Dice Score', avg_dice])
            writer.writerow(['Number of Models', len(models)])
        
        print(f"\nEnsemble Test Results:")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Dice Score: {avg_dice:.4f}")
        print(f"Results saved in: {results_path}")
        print(f"{'='*80}\n")
        
        return avg_dice

if __name__ == "__main__":
    swinunetr_cv = SwinUnetrCrossValidation(
        image_size=96,
        feature_size=24,
        learning_rate=1e-4,
        num_workers=8,                    # Number of parallel DataLoader workers
        prefetch_factor=None,             # Number of batches to prefetch per worker in DataLoader
        gradient_accumulation_steps=4,    # Emulate batch size of 4
        gradient_checkpointing=True,
        cudnn_checkpointing=True,
        epochs=100,                       # Epochs per fold
        n_folds=5,                        # 5-fold cross-validation
        early_stopping_patience=0)        # Stop if no improvement
    
    # Run 5-fold cross-validation training
    results_base = "models\\swinunetr\\results\\customswinunetr_results\\5fold_cv_AdamW_100epochs_90_10_train_val"
    fold_models, fold_results = swinunetr_cv.train_cross_validation(results_path=results_base)
    
    # Test with ensemble of all fold models
    swinunetr_cv.ensemble_test(results_path=results_base)
