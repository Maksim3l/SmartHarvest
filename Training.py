import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from PIL import Image
import json
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
import random
from datetime import datetime
import time
from collections import deque


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
    def __call__(self, val_loss, model, optimizer, scheduler, epoch, save_path):
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch, save_path)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, scheduler, epoch, save_path)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, optimizer, scheduler, epoch, save_path):
        """Saves model when validation loss decreases"""
        if self.verbose:
            print(f'  Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'best_score': self.best_score,
            'early_stopping_counter': self.counter
        }
        torch.save(checkpoint, save_path)
        self.val_loss_min = val_loss


class FruitInstanceDataset(Dataset):
    """Dataset for fruit instance segmentation using pre-generated masks"""
    
    def __init__(self, images_dir, masks_dir, transform=None, split='train', augment=True):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.split = split
        self.augment = augment and (split == 'train')
        
        # Class mapping - 5 fruits × 3 ripeness states = 15 classes + background
        self.class_map = {
            'apple': {'ripe': 1, 'unripe': 2, 'spoiled': 3},
            'cherry': {'ripe': 4, 'unripe': 5, 'spoiled': 6},
            'cucumber': {'ripe': 7, 'unripe': 8, 'spoiled': 9},
            'strawberry': {'ripe': 10, 'unripe': 11, 'spoiled': 12},
            'tomato': {'ripe': 13, 'unripe': 14, 'spoiled': 15},
        }
        
        # Fruits to skip
        self.skip_fruits = {'pear', 'pepper', 'raspberry', 'lettuce'}
        
        # Build class names for display
        self.class_names = ['background']
        for plant, ripeness_dict in self.class_map.items():
            for ripeness in ['ripe', 'unripe', 'spoiled']:
                self.class_names.append(f"{plant}-{ripeness}")
        
        self.samples = []
        self._load_samples()
        
        print(f"Dataset initialized: {len(self.samples)} samples for {split}")
        print(f"Allowed fruits: {', '.join(self.class_map.keys())}")
        print(f"Augmentation: {'Enabled' if self.augment else 'Disabled'}")
        
    def _load_samples(self):
        """Load all valid samples from mask directories"""
        semantic_dir = os.path.join(self.masks_dir, 'semantic_masks')
        instance_dir = os.path.join(self.masks_dir, 'instance_masks')
        
        if not os.path.exists(semantic_dir) or not os.path.exists(instance_dir):
            raise ValueError(f"Mask directories not found in {self.masks_dir}")
        
        # Walk through semantic masks
        for root, dirs, files in os.walk(semantic_dir):
            for filename in files:
                if not filename.endswith('_semantic.png'):
                    continue
                    
                base_name = filename.replace('_semantic.png', '')
                
                # Get plant type from directory
                rel_path = os.path.relpath(root, semantic_dir)
                plant_type = rel_path if rel_path != '.' else 'unknown'
                
                # Skip excluded fruits
                if plant_type in self.skip_fruits:
                    continue
                
                # Only process allowed fruits
                if plant_type not in self.class_map:
                    continue
                
                # Build paths
                semantic_path = os.path.join(root, filename)
                instance_path = os.path.join(root.replace(semantic_dir, instance_dir), 
                                           f"{base_name}_instance.png")
                info_path = os.path.join(root.replace(semantic_dir, instance_dir), 
                                       f"{base_name}_instances.json")
                
                # Find original image
                image_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    for img_root, _, img_files in os.walk(self.images_dir):
                        if f"{base_name}{ext}" in img_files:
                            image_path = os.path.join(img_root, f"{base_name}{ext}")
                            break
                    if image_path:
                        break
                
                # Verify all files exist
                if (image_path and os.path.exists(image_path) and 
                    os.path.exists(semantic_path) and os.path.exists(instance_path)):
                    
                    self.samples.append({
                        'image_path': image_path,
                        'semantic_path': semantic_path,
                        'instance_path': instance_path,
                        'info_path': info_path if os.path.exists(info_path) else None,
                        'plant_type': plant_type,
                        'base_name': base_name
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load image
            image = cv2.imread(sample['image_path'])
            if image is None:
                raise ValueError(f"Failed to load image: {sample['image_path']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            # Load masks
            instance_mask = np.array(Image.open(sample['instance_path']))
            
            # Load instance info
            if sample['info_path'] and os.path.exists(sample['info_path']):
                with open(sample['info_path'], 'r') as f:
                    instance_info = json.load(f)
            else:
                raise ValueError(f"Instance info not found: {sample['info_path']}")
            
            # Parse instances
            boxes = []
            labels = []
            masks = []
            areas = []
            
            for inst in instance_info.get('instances', []):
                instance_id = inst['instance_id']
                semantic_class_id = inst['semantic_class_id']
                bbox = inst['bbox']  # [x1, y1, x2, y2]
                area = inst['area']
                
                # Skip if class not in our mapping
                if semantic_class_id > 15:
                    continue
                
                # Extract instance mask
                instance_pixels = (instance_mask == instance_id)
                
                # Skip very small instances
                if np.sum(instance_pixels) < 100:
                    continue
                
                # Validate bbox
                x1, y1, x2, y2 = bbox
                if x2 <= x1 or y2 <= y1:
                    continue
                
                boxes.append([x1, y1, x2, y2])
                labels.append(semantic_class_id)
                masks.append(instance_pixels.astype(np.uint8))
                areas.append(area)
            
            # Convert to tensors
            if len(boxes) > 0:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
                areas = torch.as_tensor(areas, dtype=torch.float32)
                iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
            else:
                # Empty sample
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                masks = torch.zeros((0, height, width), dtype=torch.uint8)
                areas = torch.zeros((0,), dtype=torch.float32)
                iscrowd = torch.zeros((0,), dtype=torch.int64)
            
            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'area': areas,
                'iscrowd': iscrowd,
                'image_id': torch.tensor([idx])
            }
            
            # Apply transforms
            if self.transform:
                image = Image.fromarray(image)
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(Image.fromarray(image))
            
            return image, target
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return empty sample on error
            empty_image = torch.zeros((3, 1200, 1200))
            empty_target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'masks': torch.zeros((0, 1200, 1200), dtype=torch.uint8),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor([idx])
            }
            return empty_image, empty_target


def get_model(num_classes, dropout_rate=0.5, pretrained_backbone=True):
    """Get Mask R-CNN model with custom number of classes and dropout"""
    # Load model with pretrained weights
    model = maskrcnn_resnet50_fpn(
        weights='DEFAULT' if pretrained_backbone else None,
        weights_backbone='DEFAULT' if pretrained_backbone else None
    )
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the box predictor with dropout
    model.roi_heads.box_predictor = FastRCNNPredictorWithDropout(
        in_features, num_classes, dropout_rate
    )
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model


class FastRCNNPredictorWithDropout(nn.Module):
    """Box predictor with dropout for regularization"""
    def __init__(self, in_channels, num_classes, dropout_rate=0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        
    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


def collate_fn(batch):
    """Custom collate function for handling batches"""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None
    return tuple(zip(*batch))


class ModelCheckpointer:
    """Handles model checkpointing and recovery"""
    def __init__(self, save_dir, max_checkpoints=3):
        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.checkpoints = deque(maxlen=max_checkpoints)
        os.makedirs(save_dir, exist_ok=True)
        
    def save_checkpoint(self, model, optimizer, scheduler, epoch, train_loss, val_loss, 
                       early_stopping_state, config, is_best=False):
        """Save a checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'early_stopping_state': early_stopping_state,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save regular checkpoint
        checkpoint_name = f'checkpoint_epoch_{epoch:03d}.pth'
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        torch.save(checkpoint, checkpoint_path)
        
        # Manage checkpoint history
        self.checkpoints.append(checkpoint_path)
        if len(self.checkpoints) > self.max_checkpoints:
            # Remove oldest checkpoint (except best_model.pth)
            old_checkpoint = self.checkpoints[0]
            if os.path.exists(old_checkpoint) and 'best_model' not in old_checkpoint:
                os.remove(old_checkpoint)
                
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load a checkpoint"""
        if not os.path.exists(checkpoint_path):
            return None
        return torch.load(checkpoint_path, map_location='cpu')
    
    def find_latest_checkpoint(self):
        """Find the most recent checkpoint"""
        checkpoints = []
        for file in os.listdir(self.save_dir):
            if file.startswith('checkpoint_epoch_') and file.endswith('.pth'):
                epoch = int(file.split('_')[2].split('.')[0])
                checkpoints.append((epoch, os.path.join(self.save_dir, file)))
        
        if not checkpoints:
            return None
        
        # Sort by epoch and return latest
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1][1]


def train_model(config, resume_from=None):
    """Main training function with anti-overfitting techniques"""
    print("\n" + "="*50)
    print("FRUIT INSTANCE SEGMENTATION TRAINING")
    print("="*50)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Early stopping patience: {config['early_stopping_patience']}")
    print(f"Dropout rate: {config['dropout_rate']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Gradient clipping: {config['gradient_clip']}")
    print("="*50 + "\n")
    
    # Create save directory and checkpointer
    os.makedirs(config['save_dir'], exist_ok=True)
    checkpointer = ModelCheckpointer(config['save_dir'], max_checkpoints=3)
    
    # Setup transforms with more augmentation for training
    transform_train = transforms.Compose([
        transforms.ColorJitter(
            brightness=config['augmentation']['brightness'],
            contrast=config['augmentation']['contrast'],
            saturation=config['augmentation']['saturation'],
            hue=config['augmentation']['hue']
        ),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    
    # Check if using augmented dataset
    if config.get('use_augmented_dataset', False) and os.path.exists('AugmentedDataset/'):
        print("Using augmented dataset...")
        images_dir = 'AugmentedDataset/images/'
        masks_dir = 'AugmentedDataset/'
    else:
        images_dir = config['images_dir']
        masks_dir = config['masks_dir']
    
    full_dataset = FruitInstanceDataset(
        images_dir=images_dir,
        masks_dir=masks_dir,
        transform=None,
        augment=True
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use a fixed seed for reproducible splits
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_dataset), generator=generator)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train and val datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Apply transforms
    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders with smaller batch size to prevent overfitting
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize model with dropout
    print("\nInitializing model with dropout...")
    model = get_model(
        config['num_classes'], 
        dropout_rate=config['dropout_rate'],
        pretrained_backbone=True
    )
    model.to(config['device'])
    
    # Setup optimizer with weight decay
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=config['learning_rate'],
        momentum=0.9,
        weight_decay=config['weight_decay']  # L2 regularization
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < config['warmup_epochs']:
            return (epoch + 1) / config['warmup_epochs']
        else:
            return config['lr_gamma'] ** ((epoch - config['warmup_epochs']) // config['lr_step_size'])
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Initialize training variables
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_delta=config['early_stopping_delta'],
        verbose=True
    )
    
    # Resume from checkpoint if specified
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        checkpoint = checkpointer.load_checkpoint(resume_from)
        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            
            # Restore early stopping state
            if 'early_stopping_state' in checkpoint:
                early_stopping.counter = checkpoint['early_stopping_state'].get('counter', 0)
                early_stopping.best_score = checkpoint['early_stopping_state'].get('best_score', None)
                early_stopping.val_loss_min = checkpoint['early_stopping_state'].get('val_loss_min', np.Inf)
            
            print(f"  Resumed from epoch {start_epoch}")
            print(f"  Best val loss: {best_val_loss:.4f}")
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            if images is None:
                continue
            
            try:
                # Move to device
                images = list(img.to(config['device']) for img in images)
                targets = [{k: v.to(config['device']) for k, v in t.items()} for t in targets]
                
                # Skip batch if no valid targets
                if any(len(t['boxes']) == 0 for t in targets):
                    continue
                
                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Check for valid loss
                if not torch.isfinite(losses):
                    print(f"Warning: Non-finite loss detected, skipping batch")
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=config['gradient_clip']
                )
                
                optimizer.step()
                
                train_loss += losses.item()
                num_batches += 1
                
                # Print progress
                if batch_idx % config['print_freq'] == 0:
                    loss_components = {k: v.item() for k, v in loss_dict.items()}
                    print(f"  Batch [{batch_idx}/{len(train_loader)}] "
                          f"Loss: {losses.item():.4f} "
                          f"Components: {loss_components}")
                    
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                if images is None:
                    continue
                
                try:
                    images = list(img.to(config['device']) for img in images)
                    targets = [{k: v.to(config['device']) for k, v in t.items()} for t in targets]
                    
                    # Skip batch if no valid targets
                    if any(len(t['boxes']) == 0 for t in targets):
                        continue
                    
                    # Get loss (model needs to be in train mode for loss computation)
                    model.train()
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    model.eval()
                    
                    if torch.isfinite(losses):
                        val_loss += losses.item()
                        val_batches += 1
                        
                except Exception as e:
                    continue
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Epoch timing
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Epoch Time: {epoch_time:.1f}s")
        
        # Early stopping check
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        # Save checkpoint
        early_stopping_state = {
            'counter': early_stopping.counter,
            'best_score': early_stopping.best_score,
            'val_loss_min': early_stopping.val_loss_min
        }
        
        checkpoint_path = checkpointer.save_checkpoint(
            model, optimizer, lr_scheduler, epoch, 
            avg_train_loss, avg_val_loss, early_stopping_state, 
            config, is_best=is_best
        )
        
        # Check early stopping
        early_stopping(avg_val_loss, model, optimizer, lr_scheduler, epoch, 
                      os.path.join(config['save_dir'], 'early_stop_best.pth'))
        
        if early_stopping.early_stop:
            print("\n Early stopping triggered!")
            print(f"Best validation loss: {early_stopping.val_loss_min:.4f}")
            break
        
        # Save training curves periodically
        if (epoch + 1) % 5 == 0:
            save_training_curves(train_losses, val_losses, config['save_dir'])
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\n Training completed in {total_time/3600:.2f} hours!")
    
    # Save final model
    final_path = os.path.join(config['save_dir'], 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f" Final model saved to: {final_path}")
    
    # Save final training curves
    save_training_curves(train_losses, val_losses, config['save_dir'])
    
    # Save training summary
    summary = {
        'config': config,
        'total_epochs': len(train_losses),
        'best_val_loss': best_val_loss,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1] if val_losses else None,
        'early_stopped': early_stopping.early_stop,
        'total_time_hours': total_time / 3600
    }
    
    summary_path = os.path.join(config['save_dir'], 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return model


def save_training_curves(train_losses, val_losses, save_dir):
    """Save training curves"""
    plt.figure(figsize=(15, 5))
    
    # Loss curves
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Val Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Log scale loss
    plt.subplot(1, 3, 2)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Val Loss', color='red', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Training Loss (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Overfitting detection
    plt.subplot(1, 3, 3)
    if len(train_losses) > 1:
        overfitting_gap = [val - train for train, val in zip(train_losses, val_losses)]
        plt.plot(overfitting_gap, label='Val-Train Gap', color='purple', linewidth=2)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Validation - Training Loss')
        plt.title('Overfitting Detection')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Training curves saved to: {plot_path}")


def test_model(model_path, test_image_path, device='cuda', confidence_threshold=0.5):
    """Test the trained model on a single image"""
    print("\nTesting model on image...")
    
    # Load model
    model = get_model(num_classes=16, dropout_rate=0.0)  # No dropout for inference
    
    # Load checkpoint or state dict
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Error: Could not load image {test_image_path}")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(Image.fromarray(image))
    
    # Make prediction
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])[0]
    
    # Filter predictions by confidence
    keep = prediction['scores'] > confidence_threshold
    
    boxes = prediction['boxes'][keep].cpu()
    labels = prediction['labels'][keep].cpu()
    scores = prediction['scores'][keep].cpu()
    masks = prediction['masks'][keep].cpu()
    
    print(f"Detected {len(boxes)} objects with confidence > {confidence_threshold}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Detection results
    axes[1].imshow(original_image)
    
    # Class names mapping
    class_names = ['background',
                   'apple-ripe', 'apple-unripe', 'apple-spoiled',
                   'cherry-ripe', 'cherry-unripe', 'cherry-spoiled',
                   'cucumber-ripe', 'cucumber-unripe', 'cucumber-spoiled',
                   'strawberry-ripe', 'strawberry-unripe', 'strawberry-spoiled',
                   'tomato-ripe', 'tomato-unripe', 'tomato-spoiled']
    
    # Color map for different fruits
    fruit_colors = {
        'apple': 'red', 'cherry': 'darkred', 'cucumber': 'green',
        'strawberry': 'pink', 'tomato': 'orange'
    }
    
    # Draw predictions
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i].item()
        score = scores[i].item()
        mask = masks[i, 0].numpy()
        
        # Get fruit type for color
        class_name = class_names[label]
        fruit_type = class_name.split('-')[0]
        color = fruit_colors.get(fruit_type, 'blue')
        
        # Draw bounding box
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor=color, linewidth=2)
        axes[1].add_patch(rect)
        
        # Apply mask overlay
        masked = np.ma.masked_where(mask < 0.5, mask)
        axes[1].imshow(masked, alpha=0.3, cmap=plt.cm.colors.ListedColormap([color]))
        
        # Add label
        label_text = f"{class_name}: {score:.2f}"
        axes[1].text(x1, y1-5, label_text, 
                    bbox=dict(facecolor=color, alpha=0.7),
                    fontsize=10, color='white', weight='bold')
    
    axes[1].set_title(f'Detections (threshold={confidence_threshold})')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save result
    result_path = test_image_path.replace('.', '_result.')
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    print(f"Result saved to: {result_path}")
    
    plt.show()
    
    return prediction


def analyze_model_performance(model_path, val_loader, device='cuda', num_samples=50):
    """Analyze model performance on validation set"""
    print("\nAnalyzing model performance...")
    
    # Load model
    model = get_model(num_classes=16, dropout_rate=0.0)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Collect predictions
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader):
            if idx >= num_samples:
                break
            
            if images is None:
                continue
            
            try:
                images = list(img.to(device) for img in images)
                predictions = model(images)
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
                
            except Exception as e:
                continue

    class_counts = {}
    confidence_scores = []
    
    for pred, target in zip(all_predictions, all_targets):
        # Get predictions above threshold
        keep = pred['scores'] > 0.5
        labels = pred['labels'][keep].cpu().numpy()
        scores = pred['scores'][keep].cpu().numpy()
        
        for label, score in zip(labels, scores):
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
            confidence_scores.append(score)
    
    # Analysis results
    print("\nDetection Statistics:")
    print(f"Total detections: {len(confidence_scores)}")
    print(f"Average confidence: {np.mean(confidence_scores):.3f}")
    print(f"Confidence range: [{np.min(confidence_scores):.3f}, {np.max(confidence_scores):.3f}]")
    
    print("\nDetections per class:")
    class_names = ['background',
                   'apple-ripe', 'apple-unripe', 'apple-spoiled',
                   'cherry-ripe', 'cherry-unripe', 'cherry-spoiled',
                   'cucumber-ripe', 'cucumber-unripe', 'cucumber-spoiled',
                   'strawberry-ripe', 'strawberry-unripe', 'strawberry-spoiled',
                   'tomato-ripe', 'tomato-unripe', 'tomato-spoiled']
    
    for class_id, count in sorted(class_counts.items()):
        if class_id < len(class_names):
            print(f"  {class_names[class_id]}: {count}")
    
    return all_predictions, all_targets


def main():
    """Main execution function"""
    config = {
        'images_dir': 'RePictures/',
        'masks_dir': 'ReInstanceMasks/',
        'save_dir': 'fruit_detection_model_enhanced',
        'use_augmented_dataset': True,  # Use augmented dataset from MakeMaskInstanceSemanticDataset.py
        
        # Model parameters
        'num_classes': 16,  # 1 background + 15 fruit-ripeness combinations
        'dropout_rate': 0.3,  # Dropout for regularization
        
        # Training parameters
        'batch_size': 2,
        'learning_rate': 0.002,
        'num_epochs': 100,
        'weight_decay': 0.0005,
        'gradient_clip': 10.0,
        
        # Learning rate schedule
        'lr_step_size': 20,
        'lr_gamma': 0.5,
        'warmup_epochs': 5,
        
        # Early stopping
        'early_stopping_patience': 15,
        'early_stopping_delta': 0.0001,
        
        # Data augmentation
        'augmentation': {
            'brightness': 0.3,
            'contrast': 0.3,
            'saturation': 0.3,
            'hue': 0.1
        },
        
        # System
        'num_workers': 4,
        'print_freq': 10,
        'checkpoint_freq': 5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Check if directories exist
    if not os.path.exists(config['images_dir']):
        print(f"Error: Images directory '{config['images_dir']}' not found!")
        return
    
    if not os.path.exists(config['masks_dir']):
        print(f"Error: Masks directory '{config['masks_dir']}' not found!")
        return
    
    # Check for resume
    resume_checkpoint = None
    if os.path.exists(config['save_dir']):
        checkpointer = ModelCheckpointer(config['save_dir'])
        latest_checkpoint = checkpointer.find_latest_checkpoint()
        
        if latest_checkpoint:
            print(f"\nFound existing checkpoint: {latest_checkpoint}")
            resume = input("Resume from this checkpoint? (y/n): ").lower().strip()
            if resume == 'y':
                resume_checkpoint = latest_checkpoint
    
    # Train the model
    try:
        model = train_model(config, resume_from=resume_checkpoint)
        print("\n Training completed successfully!")
        
        test_images = []
        search_dir = 'AugmentedDataset/images/' if config['use_augmented_dataset'] else config['images_dir']
        
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')) and 'original' in file:
                    test_images.append(os.path.join(root, file))
                    if len(test_images) >= 5:
                        break
        
        if test_images:
            print(f"\nTesting on {len(test_images)} sample images...")
            model_path = os.path.join(config['save_dir'], 'best_model.pth')
            
            for test_img in test_images[:3]:
                print(f"\nTesting: {os.path.basename(test_img)}")
                test_model(model_path, test_img, config['device'])
                
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        print("You can resume training by running the script again")
    except Exception as e:
        print(f"\n Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()