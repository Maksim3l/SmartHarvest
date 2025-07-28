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


class FruitInstanceDataset(Dataset):
    """Dataset for fruit instance segmentation using pre-generated masks"""
    
    def __init__(self, images_dir, masks_dir, transform=None, split='train'):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.split = split
        
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


def get_model(num_classes):
    """Get Mask R-CNN model with custom number of classes"""
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    
    # Replace the box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    
    return model


def collate_fn(batch):
    """Custom collate function for handling batches"""
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None
    return tuple(zip(*batch))


def train_model(config):
    """Main training function"""
    print("\n" + "="*50)
    print("FRUIT INSTANCE SEGMENTATION TRAINING")
    print("="*50)
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Fruits: apple, cherry, cucumber, strawberry, tomato")
    print(f"Excluded: pear, pepper, raspberry, lettuce")
    print("="*50 + "\n")
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Setup transforms
    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Loading datasets...")
    full_dataset = FruitInstanceDataset(
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        transform=None  # We'll apply transforms after split
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(config['num_classes'])
    model.to(config['device'])
    
    # Setup optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=config['learning_rate'],
        momentum=0.9,
        weight_decay=0.0005
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['lr_step_size'],
        gamma=config['lr_gamma']
    )
    
    # Training variables
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(config['num_epochs']):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                optimizer.step()
                
                train_loss += losses.item()
                num_batches += 1
                
                # Print progress
                if batch_idx % config['print_freq'] == 0:
                    print(f"  Batch [{batch_idx}/{len(train_loader)}] Loss: {losses.item():.4f}")
                    
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
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 
                      os.path.join(config['save_dir'], 'best_model.pth'))
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % config['checkpoint_freq'] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config
            }
            checkpoint_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(config['save_dir'], 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Training completed! Final model saved to: {final_path}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(config['save_dir'], 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Training curves saved to: {plot_path}")
    plt.show()
    
    return model


def test_model(model_path, test_image_path, device='cuda'):
    """Test the trained model on a single image"""
    print("\nTesting model on image...")
    
    # Load model
    model = get_model(num_classes=16)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(Image.fromarray(image))
    
    # Make prediction
    with torch.no_grad():
        prediction = model([image_tensor.to(device)])[0]
    
    # Filter predictions
    score_threshold = 0.5
    keep = prediction['scores'] > score_threshold
    
    boxes = prediction['boxes'][keep].cpu()
    labels = prediction['labels'][keep].cpu()
    scores = prediction['scores'][keep].cpu()
    masks = prediction['masks'][keep].cpu()
    
    print(f"Detected {len(boxes)} objects")
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    # Class names mapping
    class_names = ['background',
                   'apple-ripe', 'apple-unripe', 'apple-spoiled',
                   'cherry-ripe', 'cherry-unripe', 'cherry-spoiled',
                   'cucumber-ripe', 'cucumber-unripe', 'cucumber-spoiled',
                   'strawberry-ripe', 'strawberry-unripe', 'strawberry-spoiled',
                   'tomato-ripe', 'tomato-unripe', 'tomato-spoiled']
    
    # Draw predictions
    for i in range(len(boxes)):
        box = boxes[i]
        label = labels[i].item()
        score = scores[i].item()
        
        # Draw bounding box
        x1, y1, x2, y2 = box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Add label
        label_text = f"{class_names[label]}: {score:.2f}"
        plt.text(x1, y1-5, label_text, 
                bbox=dict(facecolor='red', alpha=0.5),
                fontsize=10, color='white')
    
    plt.axis('off')
    plt.title('Fruit Detection Results')
    plt.tight_layout()
    plt.show()
    
    return prediction


def main():
    """Main execution function"""
    # Training configuration
    config = {
        'images_dir': 'RePictures/',          # Your resized images
        'masks_dir': 'ReInstanceMasks/',      # Your pre-generated masks
        'save_dir': 'fruit_detection_model',
        'num_classes': 16,                    # 1 background + 15 fruit-ripeness combinations
        'batch_size': 4,
        'learning_rate': 0.005,
        'num_epochs': 50,
        'lr_step_size': 15,
        'lr_gamma': 0.1,
        'num_workers': 4,
        'print_freq': 10,
        'checkpoint_freq': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Check if directories exist
    if not os.path.exists(config['images_dir']):
        print(f"Error: Images directory '{config['images_dir']}' not found!")
        print("Please run the mask generation script first.")
        return
    
    if not os.path.exists(config['masks_dir']):
        print(f"Error: Masks directory '{config['masks_dir']}' not found!")
        print("Please run the mask generation script first.")
        return
    
    # Train the model
    try:
        model = train_model(config)
        print("\n✅ Training completed successfully!")
        
        # Test on a sample image if available
        test_images = []
        for root, dirs, files in os.walk(config['images_dir']):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(root, file))
                    if len(test_images) >= 5:  # Get a few test images
                        break
        
        if test_images:
            print(f"\nTesting on {len(test_images)} sample images...")
            model_path = os.path.join(config['save_dir'], 'best_model.pth')
            for test_img in test_images[:3]:  # Test on first 3
                print(f"\nTesting: {test_img}")
                test_model(model_path, test_img, config['device'])
                
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()