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
import torch.nn.functional as F


class FruitInstanceDataset(Dataset):
    def __init__(self, json_file, images_dir, transform=None, target_size=(1200, 1200), include_negatives=True):
        self.images_dir = images_dir
        self.transform = transform
        self.target_size = target_size
        self.include_negatives = include_negatives

        with open(json_file, 'r') as f:
            self.annotations = json.load(f)

        self.class_map = {
            'apple': {'ripe': 1, 'unripe': 2, 'spoiled': 3},
            'cherry': {'ripe': 4, 'unripe': 5, 'spoiled': 6},
            'cucumber': {'ripe': 7, 'unripe': 8, 'spoiled': 9},
            'lettuce': {'ripe': 10, 'unripe': 11, 'spoiled': 12},
            'pear': {'ripe': 13, 'unripe': 14, 'spoiled': 15},
            'pepper': {'ripe': 16, 'unripe': 17, 'spoiled': 18},
            'raspberry': {'ripe': 19, 'unripe': 20, 'spoiled': 21},
            'strawberry': {'ripe': 22, 'unripe': 23, 'spoiled': 24},
            'tomato': {'ripe': 25, 'unripe': 26, 'spoiled': 27},
        }

        self.class_names = ['background']
        for plant, ripeness_dict in self.class_map.items():
            for ripeness, class_id in ripeness_dict.items():
                self.class_names.append(f"{plant}-{ripeness}")

        # Process all samples - both positive (with annotations) and negative (background)
        self.samples = []
        positive_samples = 0
        negative_samples = 0
        
        for key, image_data in self.annotations.items():
            if (isinstance(image_data, dict) and 
                'filename' in image_data and 
                'regions' in image_data):
        
                plant = image_data.get('file_attributes', {}).get('plant', '').lower()
                regions = image_data['regions']
        
                # Process regions and keep only valid ones
                valid_regions = []
                for region in regions:
                    if (region and 
                        isinstance(region, dict) and
                        'shape_attributes' in region and
                        'region_attributes' in region and
                        region.get('shape_attributes', {}).get('name') == 'polygon'):
                        
                        x_points = region['shape_attributes'].get('all_points_x', [])
                        y_points = region['shape_attributes'].get('all_points_y', [])
                        
                        if len(x_points) >= 3 and len(y_points) >= 3:
                            ripeness = region.get('region_attributes', {}).get('ripeness_factor', 'ripe')
                            if plant in self.class_map and ripeness in self.class_map[plant]:
                                valid_regions.append(region)
        
                # Store the cleaned version
                cleaned_data = image_data.copy()
                cleaned_data['regions'] = valid_regions
                cleaned_data['is_negative'] = len(valid_regions) == 0
                
                # Always include the sample (positive or negative)
                if len(valid_regions) > 0:
                    positive_samples += 1
                else:
                    negative_samples += 1
                    
                if self.include_negatives or len(valid_regions) > 0:
                    self.samples.append(cleaned_data)

        print(f"Dataset loaded:")
        print(f"  - Positive samples (with annotations): {positive_samples}")
        print(f"  - Negative samples (background only): {negative_samples}")
        print(f"  - Total samples: {len(self.samples)}")
        
        if not self.include_negatives:
            print(f"  - Using only positive samples for training")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                current_idx = (idx + attempt) % len(self.samples)
                image_data = self.samples[current_idx]
                filename = image_data['filename']
                plant_type = image_data.get('file_attributes', {}).get('plant', 'unknown').lower()
                regions = image_data['regions']
                is_negative = image_data.get('is_negative', False)
                
                image_path = self.find_image_file(filename)
                if not image_path:
                    if attempt == max_retries - 1:
                        raise ValueError(f"Image not found: {filename}")
                    continue
        
                image = cv2.imread(image_path)
                if image is None:
                    if attempt == max_retries - 1:
                        raise ValueError(f"Could not load image: {image_path}")
                    continue
                    
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                height, width = image.shape[:2]

                boxes = []
                labels = []
                masks = []
                areas = []

                # Process annotations (if any)
                for region in regions:
                    if not region or not isinstance(region, dict):
                        continue
                        
                    shape_attrs = region.get('shape_attributes', {})
                    if shape_attrs.get('name') != 'polygon':
                        continue

                    x_points = shape_attrs.get('all_points_x', [])
                    y_points = shape_attrs.get('all_points_y', [])

                    if len(x_points) < 3 or len(y_points) < 3 or len(x_points) != len(y_points):
                        continue

                    # Ensure coordinates are within image bounds
                    x_points = [max(0, min(width-1, x)) for x in x_points]
                    y_points = [max(0, min(height-1, y)) for y in y_points]

                    polygon_coords = [(x, y) for x, y in zip(x_points, y_points)]
                    mask = np.zeros((height, width), dtype=np.uint8)
                    
                    try:
                        cv2.fillPoly(mask, [np.array(polygon_coords, dtype=np.int32)], 1)
                    except Exception as e:
                        continue
                        
                    area = np.sum(mask)
                    if area < 100:  # Skip very small masks
                        continue

                    pos = np.where(mask)
                    if len(pos[0]) == 0:
                        continue

                    xmin, xmax = np.min(pos[1]), np.max(pos[1])
                    ymin, ymax = np.min(pos[0]), np.max(pos[0])

                    # Ensure valid bounding box
                    if xmax <= xmin or ymax <= ymin or (xmax - xmin) < 5 or (ymax - ymin) < 5:
                        continue

                    ripeness = region.get('region_attributes', {}).get('ripeness_factor', 'ripe')
                
                    if plant_type in self.class_map and ripeness in self.class_map[plant_type]:
                        label = self.class_map[plant_type][ripeness]
                    else:
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)
                    masks.append(mask)
                    areas.append(area)

                # Create proper empty tensors for negative samples (background images)
                if len(boxes) == 0:
                    # This is a negative sample - no objects present
                    boxes = torch.zeros((0, 4), dtype=torch.float32)
                    labels = torch.zeros((0,), dtype=torch.int64)
                    masks = torch.zeros((0, height, width), dtype=torch.uint8)
                    areas = torch.zeros((0,), dtype=torch.float32)
                    iscrowd = torch.zeros((0,), dtype=torch.int64)
                else:
                    # This is a positive sample - convert to tensors
                    boxes = torch.as_tensor(boxes, dtype=torch.float32)
                    labels = torch.as_tensor(labels, dtype=torch.int64)
                    masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
                    areas = torch.as_tensor(areas, dtype=torch.float32)
                    iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

                target = {
                    'boxes': boxes,
                    'labels': labels,
                    'masks': masks,
                    'area': areas,
                    'iscrowd': iscrowd,
                    'image_id': torch.tensor([current_idx]),
                    'is_negative': is_negative  # Flag for tracking
                }
        
                if self.transform:
                    image = Image.fromarray(image)
                    image = self.transform(image)
                else:
                    image = transforms.ToTensor()(Image.fromarray(image))
        
                return image, target
                
            except Exception as e:
                print(f"Error processing sample {current_idx} (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise e
                continue
    
    def find_image_file(self, filename):
        direct_path = os.path.join(self.images_dir, filename)
        if os.path.exists(direct_path):
            return direct_path
        for root, _, files in os.walk(self.images_dir):
            if filename in files:
                return os.path.join(root, filename)
        return None


def get_instance_model(num_classes):
    """Create Mask R-CNN model with custom number of classes."""
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def collate_fn(batch):
    """Custom collate function that properly handles both positive and negative samples."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None
    return tuple(zip(*batch))


def train_instance_model():
    """Main training function for the instance segmentation model."""
    config = {
        'batch_size': 4,
        'learning_rate': 5e-4,
        'num_epochs': 50,
        'num_classes': 28,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'instance_checkpoints',
        'warmup_epochs': 5,
        'print_freq': 10,
        'include_negatives': True
    }

    print(f"Training on device: {config['device']}")

    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((1200, 1200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset
    dataset = FruitInstanceDataset(
        json_file="via_export_json(4).json",
        images_dir="Pictures/",
        transform=transform,
        include_negatives=config['include_negatives']
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(73)
    )

    print(f"Training set: {len(train_dataset)} images")
    print(f"Validation set: {len(val_dataset)} images")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    # Initialize model
    model = get_instance_model(config['num_classes'])
    model.to(config['device'])

    # Setup optimizer with different learning rates for backbone and head
    params = [
        {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': config['learning_rate'] * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': config['learning_rate']}
    ]

    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 35], gamma=0.1)

    # Training tracking
    train_losses = []
    val_losses = []
    os.makedirs(config['save_dir'], exist_ok=True)

    # Training loop
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print("-" * 50)

        model.train()
        epoch_loss = 0
        num_batches = 0
        positive_batches = 0
        negative_batches = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            if images is None or targets is None:
                continue

            try:
                images = [img.to(config['device']) for img in images]
                targets = [{k: v.to(config['device']) if torch.is_tensor(v) else v 
                           for k, v in t.items()} for t in targets]
                
                # Count positive vs negative samples in batch
                batch_negatives = sum(1 for t in targets if t.get('is_negative', False))
                batch_positives = len(targets) - batch_negatives
                
                positive_batches += batch_positives
                negative_batches += batch_negatives
                
                # Forward pass - MaskRCNN handles empty targets automatically
                loss_dict = model(images, targets)
                
                if not isinstance(loss_dict, dict) or len(loss_dict) == 0:
                    continue
                
                losses = sum(loss for loss in loss_dict.values())
                
                if not torch.isfinite(losses):
                    print(f"Warning: Non-finite loss at batch {batch_idx}, skipping")
                    continue
                
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += losses.item()
                num_batches += 1

                if batch_idx % config['print_freq'] == 0:
                    print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {losses.item():.4f}")
                    print(f"  Positive samples: {batch_positives}, Negative samples: {batch_negatives}")
                    for k, v in loss_dict.items():
                        print(f"  {k}: {v.item():.4f}")
                        
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        if num_batches == 0:
            print("Warning: No valid batches processed in this epoch!")
            continue
            
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        print(f"Epoch {epoch+1} training summary:")
        print(f"  Total positive samples: {positive_batches}")
        print(f"  Total negative samples: {negative_batches}")

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for images, targets in val_loader:
                if images is None or targets is None:
                    continue
                    
                try:
                    images = [img.to(config['device']) for img in images]
                    targets = [{k: v.to(config['device']) if torch.is_tensor(v) else v 
                               for k, v in t.items()} for t in targets]
                    
                    # Switch to train mode for loss computation
                    model.train()
                    loss_dict = model(images, targets)
                    model.eval()
                    
                    if isinstance(loss_dict, dict) and len(loss_dict) > 0:
                        losses = sum(loss for loss in loss_dict.values())
                        if torch.isfinite(losses):
                            val_loss += losses.item()
                            val_batches += 1
                        
                except Exception as e:
                    continue

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_losses.append(avg_val_loss)
        scheduler.step()

        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoints
        if (epoch + 1) % 10 == 0 or epoch == config['num_epochs'] - 1:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'config': config,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            checkpoint_path = os.path.join(config['save_dir'], f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(config['save_dir'], "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['save_dir'], 'training_curves.png'))
    plt.show()

    print("Training completed!")
    print(f"Final model saved: {final_model_path}")
    return model


def generate_pseudo_labels_with_maskrcnn(model_path, unlabeled_images_dir, output_dir, 
                                       confidence_threshold=0.7, nms_threshold=0.5):
    """Generate pseudo labels using trained Mask R-CNN model."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    model = get_instance_model(28)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() 

    # Class mapping for pseudo labels
    class_map = {
        1: {'plant': 'apple', 'ripeness': 'ripe'},
        2: {'plant': 'apple', 'ripeness': 'unripe'},
        3: {'plant': 'apple', 'ripeness': 'spoiled'},
        4: {'plant': 'cherry', 'ripeness': 'ripe'},
        5: {'plant': 'cherry', 'ripeness': 'unripe'},
        6: {'plant': 'cherry', 'ripeness': 'spoiled'},
        7: {'plant': 'cucumber', 'ripeness': 'ripe'},
        8: {'plant': 'cucumber', 'ripeness': 'unripe'},
        9: {'plant': 'cucumber', 'ripeness': 'spoiled'},
        10: {'plant': 'lettuce', 'ripeness': 'ripe'},
        11: {'plant': 'lettuce', 'ripeness': 'unripe'},
        12: {'plant': 'lettuce', 'ripeness': 'spoiled'},
        13: {'plant': 'pear', 'ripeness': 'ripe'},
        14: {'plant': 'pear', 'ripeness': 'unripe'},
        15: {'plant': 'pear', 'ripeness': 'spoiled'},
        16: {'plant': 'pepper', 'ripeness': 'ripe'},
        17: {'plant': 'pepper', 'ripeness': 'unripe'},
        18: {'plant': 'pepper', 'ripeness': 'spoiled'},
        19: {'plant': 'raspberry', 'ripeness': 'ripe'},
        20: {'plant': 'raspberry', 'ripeness': 'unripe'},
        21: {'plant': 'raspberry', 'ripeness': 'spoiled'},
        22: {'plant': 'strawberry', 'ripeness': 'ripe'},
        23: {'plant': 'strawberry', 'ripeness': 'unripe'},
        24: {'plant': 'strawberry', 'ripeness': 'spoiled'},
        25: {'plant': 'tomato', 'ripeness': 'ripe'},
        26: {'plant': 'tomato', 'ripeness': 'unripe'},
        27: {'plant': 'tomato', 'ripeness': 'spoiled'},
    }

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((1200, 1200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Setup output directories
    os.makedirs(output_dir, exist_ok=True)
    pseudo_masks_dir = os.path.join(output_dir, 'pseudo_masks')
    pseudo_annotations_dir = os.path.join(output_dir, 'pseudo_annotations')
    os.makedirs(pseudo_masks_dir, exist_ok=True)
    os.makedirs(pseudo_annotations_dir, exist_ok=True)

    processed_count = 0
    total_instances = 0
    negative_images = 0

    print("Generating pseudo labels...")

    for root, _, files in os.walk(unlabeled_images_dir):
        for filename in files:
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(root, filename)

            try:
                image = Image.open(image_path).convert('RGB')
                original_size = image.size

                with torch.no_grad():
                    image_tensor = transform(image).unsqueeze(0).to(device)
                    predictions = model(image_tensor)

                pred = predictions[0]
                keep_idx = pred['scores'] > confidence_threshold

                if torch.sum(keep_idx) == 0:
                    # This is a negative image (no detections above threshold)
                    negative_images += 1
                    
                    # Still create an annotation file for consistency
                    base_name = os.path.splitext(filename)[0]
                    pseudo_annotation = {
                        filename: {
                            'filename': filename,
                            'size': os.path.getsize(image_path),
                            'regions': [],  # Empty regions for negative sample
                            'file_attributes': {
                                'plant': 'background'  # Mark as background image
                            }
                        }
                    }
                    
                    annotation_path = os.path.join(pseudo_annotations_dir, f"{base_name}_negative.json")
                    with open(annotation_path, 'w') as f:
                        json.dump(pseudo_annotation, f, indent=2)
                    
                    processed_count += 1
                    print(f"Generated negative sample annotation for {filename}")
                    continue

                # Apply NMS for positive detections
                keep_idx = torch.ops.torchvision.nms(
                    pred['boxes'][keep_idx], 
                    pred['scores'][keep_idx], 
                    nms_threshold
                )

                boxes = pred['boxes'][keep_idx]
                labels = pred['labels'][keep_idx]
                masks = pred['masks'][keep_idx]
                scores = pred['scores'][keep_idx]

                if len(boxes) == 0:
                    negative_images += 1
                    continue

                base_name = os.path.splitext(filename)[0]
                regions = []

                height, width = original_size[1], original_size[0]
                instance_mask = np.zeros((height, width), dtype=np.uint16)
                semantic_mask = np.zeros((height, width), dtype=np.uint8)

                for i, (box, label, mask, score) in enumerate(zip(boxes, labels, masks, scores)):
                    mask_np = mask[0].cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (width, height), 
                                            interpolation=cv2.INTER_LINEAR)
                    mask_binary = mask_resized > 0.5

                    if np.sum(mask_binary) < 100:
                        continue

                    contours, _ = cv2.findContours(
                        mask_binary.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    if len(contours) == 0:
                        continue

                    largest_contour = max(contours, key=cv2.contourArea)
                    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

                    if len(simplified) < 3:
                        continue

                    x_points = [int(point[0][0]) for point in simplified]
                    y_points = [int(point[0][1]) for point in simplified]

                    class_id = label.item()
                    
                    if class_id in class_map:
                        plant_type = class_map[class_id]['plant']
                        ripeness = class_map[class_id]['ripeness']
                    else:
                        continue

                    region = {
                        'shape_attributes': {
                            'name': 'polygon',
                            'all_points_x': x_points,
                            'all_points_y': y_points
                        },
                        'region_attributes': {
                            'ripeness_factor': ripeness
                        }
                    }

                    regions.append(region)
                    instance_mask[mask_binary] = i + 1
                    semantic_mask[mask_binary] = class_id
                    total_instances += 1

                if len(regions) > 0:
                    pseudo_annotation = {
                        filename: {
                            'filename': filename,
                            'size': os.path.getsize(image_path),
                            'regions': regions,
                            'file_attributes': {
                                'plant': class_map[labels[0].item()]['plant']
                            }
                        }
                    }
                    annotation_path = os.path.join(pseudo_annotations_dir, f"{base_name}_pseudo.json")

                    with open(annotation_path, 'w') as f:
                        json.dump(pseudo_annotation, f, indent=2)

                    mask_dir = os.path.join(pseudo_masks_dir, class_map[labels[0].item()]['plant'])
                    os.makedirs(mask_dir, exist_ok=True)
                    
                    Image.fromarray(semantic_mask).save(
                        os.path.join(mask_dir, f"{base_name}_semantic.png")
                    )
                    Image.fromarray(instance_mask).save(
                        os.path.join(mask_dir, f"{base_name}_instance.png")
                    )

                processed_count += 1
                print(f"Generated pseudo labels for {filename}: {len(regions)} instances")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    print(f"\nPseudo Label Generation Summary:")
    print(f"Processed: {processed_count} images")
    print(f"Positive images (with detections): {processed_count - negative_images}")
    print(f"Negative images (background only): {negative_images}")
    print(f"Total pseudo instances: {total_instances}")
    print(f"Average instances per positive image: {total_instances/(processed_count - negative_images):.1f}" if processed_count - negative_images > 0 else 0)

    return processed_count, total_instances


def visualize_predictions(model_path, test_image_path, save_path=None):
    """
    Visualize predictions from a trained Mask R-CNN model on a test image.
    
    Args:
        model_path: Path to the trained model weights
        test_image_path: Path to the test image
        save_path: Optional path to save the visualization
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the trained model
    model = get_instance_model(28)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and preprocess the image
    image = Image.open(test_image_path).convert('RGB')
    original_image = np.array(image)

    transform = transforms.Compose([
        transforms.Resize((1200, 1200)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Make predictions
    with torch.no_grad():
        image_tensor = transform(image).unsqueeze(0).to(device)
        predictions = model(image_tensor)

    pred = predictions[0]
    confidence_threshold = 0.5

    # Filter predictions by confidence
    keep = pred['scores'] > confidence_threshold
    boxes = pred['boxes'][keep].cpu().numpy()
    labels = pred['labels'][keep].cpu().numpy()
    masks = pred['masks'][keep].cpu().numpy()
    scores = pred['scores'][keep].cpu().numpy()
    
    # Class names mapping
    class_names = ['background']
    class_map = {
        'apple': {'ripe': 1, 'unripe': 2, 'spoiled': 3},
        'cherry': {'ripe': 4, 'unripe': 5, 'spoiled': 6},
        'cucumber': {'ripe': 7, 'unripe': 8, 'spoiled': 9},
        'lettuce': {'ripe': 10, 'unripe': 11, 'spoiled': 12},
        'pear': {'ripe': 13, 'unripe': 14, 'spoiled': 15},
        'pepper': {'ripe': 16, 'unripe': 17, 'spoiled': 18},
        'raspberry': {'ripe': 19, 'unripe': 20, 'spoiled': 21},
        'strawberry': {'ripe': 22, 'unripe': 23, 'spoiled': 24},
        'tomato': {'ripe': 25, 'unripe': 26, 'spoiled': 27},
    }
    
    for plant, ripeness_dict in class_map.items():
        for ripeness, class_id in ripeness_dict.items():
            class_names.append(f"{plant}-{ripeness}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    resized_original = cv2.resize(original_image, (1200, 1200))
    axes[0].imshow(resized_original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Image with bounding boxes and labels
    image_with_boxes = resized_original.copy()
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        
        # Draw bounding box
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add label and confidence
        if label < len(class_names):
            label_text = f'{class_names[label]}: {score:.2f}'
        else:
            label_text = f'Class {label}: {score:.2f}'
            
        cv2.putText(image_with_boxes, label_text, 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    axes[1].imshow(image_with_boxes)
    axes[1].set_title(f'Predictions (conf > {confidence_threshold})')
    axes[1].axis('off')

    # Combined instance masks
    combined_mask = np.zeros((1200, 1200, 3))
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(masks), 1)))

    for i, (mask, color) in enumerate(zip(masks, colors)):
        mask_resized = cv2.resize(mask[0], (1200, 1200))
        mask_binary = mask_resized > 0.5
        combined_mask[mask_binary] = color[:3]

    axes[2].imshow(combined_mask)
    axes[2].set_title('Instance Masks')
    axes[2].axis('off')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print detection summary
    print(f"Detected {len(boxes)} instances")
    for i, (label, score) in enumerate(zip(labels, scores)):
        if label < len(class_names):
            label_name = class_names[label]
        else:
            label_name = f"Class {label}"
        print(f"Instance {i+1}: {label_name}, Confidence {score:.3f}")

    return boxes, labels, masks, scores


def batch_predict_and_visualize(model_path, image_dir, output_dir, confidence_threshold=0.5):
    """
    Run predictions on all images in a directory and save visualizations.
    
    Args:
        model_path: Path to trained model
        image_dir: Directory containing test images
        output_dir: Directory to save prediction visualizations
        confidence_threshold: Minimum confidence for detections
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_instance_model(28)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    processed = 0
    
    for filename in os.listdir(image_dir):
        if not filename.lower().endswith(image_extensions):
            continue
            
        image_path = os.path.join(image_dir, filename)
        output_path = os.path.join(output_dir, f"pred_{filename}")
        
        try:
            visualize_predictions(model_path, image_path, output_path)
            processed += 1
            print(f"Processed: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    print(f"\nBatch prediction complete. Processed {processed} images.")
    print(f"Results saved to: {output_dir}")


def evaluate_model_performance(model_path, val_dataset, device='cuda'):
    """
    Evaluate model performance on validation dataset.
    
    Args:
        model_path: Path to trained model
        val_dataset: Validation dataset
        device: Device to run evaluation on
    """
    model = get_instance_model(28)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    total_detections = 0
    total_ground_truth = 0
    confidence_threshold = 0.5
    
    with torch.no_grad():
        for images, targets in val_loader:
            if images is None or targets is None:
                continue
                
            images = [img.to(device) for img in images]
            predictions = model(images)
            
            for pred, target in zip(predictions, targets):
                # Count predictions above threshold
                keep = pred['scores'] > confidence_threshold
                total_detections += torch.sum(keep).item()
                
                # Count ground truth objects
                total_ground_truth += len(target['labels'])
    
    print(f"Evaluation Results:")
    print(f"Total Ground Truth Objects: {total_ground_truth}")
    print(f"Total Detections (conf > {confidence_threshold}): {total_detections}")
    print(f"Detection Rate: {total_detections/total_ground_truth:.3f}" if total_ground_truth > 0 else "N/A")


def get_class_statistics(dataset):
    """
    Get statistics about class distribution in dataset.
    
    Args:
        dataset: FruitInstanceDataset object
    """
    class_counts = {}
    total_instances = 0
    positive_images = 0
    negative_images = 0
    
    for i in range(len(dataset)):
        try:
            _, target = dataset[i]
            
            if target['is_negative']:
                negative_images += 1
            else:
                positive_images += 1
                
            labels = target['labels']
            total_instances += len(labels)
            
            for label in labels:
                label_item = label.item()
                if label_item in class_counts:
                    class_counts[label_item] += 1
                else:
                    class_counts[label_item] = 1
                    
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Dataset Statistics:")
    print(f"Positive images (with annotations): {positive_images}")
    print(f"Negative images (background only): {negative_images}")
    print(f"Total instances: {total_instances}")
    print(f"Average instances per positive image: {total_instances/positive_images:.2f}" if positive_images > 0 else "N/A")
    
    print(f"\nClass Distribution:")
    for class_id, count in sorted(class_counts.items()):
        if class_id < len(dataset.class_names):
            class_name = dataset.class_names[class_id]
        else:
            class_name = f"Class {class_id}"
        print(f"  {class_name}: {count} instances")


def main():
    """Main execution pipeline."""
    print("=== FRUIT INSTANCE SEGMENTATION TRAINING PIPELINE ===")
    
    # Step 1: Training
    print("\n1. Training Instance Segmentation Model...")
    try:
        trained_model = train_instance_model()
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return

    # Step 2: Pseudo Label Generation
    print("\n2. Generating Pseudo Labels...")
    model_path = "instance_checkpoints/final_model.pth"
    unlabeled_dir = "unlabeled_images/" 
    pseudo_output_dir = "pseudo_labels/"
    
    if os.path.exists(model_path) and os.path.exists(unlabeled_dir):
        try:
            processed, total = generate_pseudo_labels_with_maskrcnn(
                model_path, unlabeled_dir, pseudo_output_dir, 
                confidence_threshold=0.7
            )
            print(f"✓ Generated pseudo labels for {processed} images with {total} instances")
        except Exception as e:
            print(f"✗ Pseudo label generation failed: {e}")
    else:
        print("⚠ Skipping pseudo label generation - model or unlabeled images not found")

    # Step 3: Test Prediction Visualization
    print("\n3. Test Prediction Visualization...")
    test_image = "Pictures/Cherry/CA-cherries-split2.jpg"
    
    if os.path.exists(model_path) and os.path.exists(test_image):
        try:
            visualize_predictions(model_path, test_image, "prediction_visualization.png")
            print("✓ Visualization completed")
        except Exception as e:
            print(f"✗ Visualization failed: {e}")
    else:
        print("⚠ Skipping visualization - model or test image not found")
    
    # Step 4: Dataset Statistics (Optional)
    print("\n4. Dataset Analysis...")
    try:
        transform = transforms.Compose([
            transforms.Resize((1200, 1200)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        dataset = FruitInstanceDataset(
            json_file="via_export_json(4).json",
            images_dir="Pictures/",
            transform=transform,
            include_negatives=True
        )
        
        get_class_statistics(dataset)
        print("✓ Dataset analysis completed")
    except Exception as e:
        print(f"✗ Dataset analysis failed: {e}")
    
    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()