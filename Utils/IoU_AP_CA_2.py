import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import numpy as np
from PIL import Image
import json
import cv2
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import warnings
import gc
import re
warnings.filterwarnings('ignore')

# Import your existing classes
from Training import FruitInstanceDataset, get_model, FastRCNNPredictorWithDropout, collate_fn


class OriginalImagesDataset(FruitInstanceDataset):
    """Dataset that only loads original (non-augmented) images for evaluation"""
    
    def __init__(self, images_dir, masks_dir, transform=None, target_fruits=['apple', 'cherry']):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.target_fruits = target_fruits
        
        # Class mapping for apple and cherry only (matching your training)
        self.class_map = {
            'apple': {'ripe': 1, 'unripe': 2, 'spoiled': 3},
            'cherry': {'ripe': 4, 'unripe': 5, 'spoiled': 6},
        }
        
        self.class_names = ['background']
        for plant in self.target_fruits:
            for ripeness in ['ripe', 'unripe', 'spoiled']:
                self.class_names.append(f"{plant}-{ripeness}")
        
        self.samples = []
        self._load_original_samples()
        
        print(f"Original images dataset: {len(self.samples)} samples")
        
    def _load_original_samples(self):
        """Load only original (non-augmented) samples"""
        semantic_dir = os.path.join(self.masks_dir, 'semantic_masks')
        instance_dir = os.path.join(self.masks_dir, 'instance_masks')

        if not os.path.exists(semantic_dir) or not os.path.exists(instance_dir):
            raise ValueError(f"Mask directories not found in {self.masks_dir}")

        for plant_type in self.target_fruits:
            plant_semantic_dir = os.path.join(semantic_dir, plant_type)
            plant_instance_dir = os.path.join(instance_dir, plant_type)
            plant_images_dir = os.path.join(self.images_dir, plant_type)
            
            if not os.path.exists(plant_semantic_dir):
                print(f"Warning: {plant_semantic_dir} not found")
                continue
                
            for filename in os.listdir(plant_semantic_dir):
                # Only process original files (not augmented)
                if not filename.endswith('_original_semantic.png'):
                    continue

                base_name = filename.replace('_semantic.png', '')
                image_base = base_name

                # Build paths
                semantic_path = os.path.join(plant_semantic_dir, filename)
                instance_filename = base_name + '_instance.png'
                info_filename = base_name + '_instances.json'
                
                instance_path = os.path.join(plant_instance_dir, instance_filename)
                info_path = os.path.join(plant_instance_dir, info_filename)

                # Find corresponding image file
                image_path = None
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    potential_path = os.path.join(plant_images_dir, f"{image_base}{ext}")
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break

                # Verify all files exist
                if (image_path and os.path.exists(image_path) and 
                    os.path.exists(semantic_path) and 
                    os.path.exists(instance_path) and 
                    os.path.exists(info_path)):
                    
                    self.samples.append({
                        'image_path': image_path,
                        'semantic_path': semantic_path,
                        'instance_path': instance_path,
                        'info_path': info_path,
                        'plant_type': plant_type,
                        'base_name': base_name
                    })


def calculate_iou_fast(box1, box2):
    """Fast IoU calculation"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def apply_nms_efficient(prediction, nms_threshold=0.5):
    """Memory-efficient NMS application"""
    if len(prediction['boxes']) == 0:
        return prediction
    
    keep_indices = []
    
    # Process each class separately
    unique_labels = torch.unique(prediction['labels'])
    
    for class_id in unique_labels:
        class_mask = prediction['labels'] == class_id
        class_indices = torch.where(class_mask)[0]
        
        if len(class_indices) > 0:
            class_boxes = prediction['boxes'][class_indices]
            class_scores = prediction['scores'][class_indices]
            
            # Apply NMS
            keep = nms(class_boxes, class_scores, nms_threshold)
            keep_indices.extend(class_indices[keep].tolist())
    
    if keep_indices:
        keep_indices = torch.tensor(keep_indices, dtype=torch.long)
        
        # Sort by score
        scores = prediction['scores'][keep_indices]
        sorted_indices = torch.argsort(scores, descending=True)
        keep_indices = keep_indices[sorted_indices]
        
        return {
            'boxes': prediction['boxes'][keep_indices],
            'labels': prediction['labels'][keep_indices],
            'scores': prediction['scores'][keep_indices],
            'masks': prediction['masks'][keep_indices] if 'masks' in prediction else None
        }
    else:
        return {
            'boxes': torch.empty((0, 4), dtype=torch.float32),
            'labels': torch.empty((0,), dtype=torch.long),
            'scores': torch.empty((0,), dtype=torch.float32),
            'masks': torch.empty((0, 1, 1), dtype=torch.uint8) if 'masks' in prediction else None
        }


class StreamingEvaluator:
    """Memory-efficient streaming evaluator"""
    
    def __init__(self, num_classes, class_names, iou_thresholds=[0.5, 0.75]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.iou_thresholds = iou_thresholds
        
        # Storage for streaming evaluation
        self.results = {iou_thresh: {
            'tp_scores': defaultdict(list),  # true positive scores per class
            'fp_scores': defaultdict(list),  # false positive scores per class
            'gt_counts': defaultdict(int)    # ground truth counts per class
        } for iou_thresh in iou_thresholds}
    
    def process_batch(self, predictions, targets, confidence_threshold=0.5, nms_threshold=0.5):
        """Process a single batch and update running statistics"""
        
        for pred, target in zip(predictions, targets):
            # Apply NMS
            pred = apply_nms_efficient(pred, nms_threshold)
            
            # Filter by confidence
            if len(pred['scores']) > 0:
                keep = pred['scores'] > confidence_threshold
                filtered_pred = {
                    'boxes': pred['boxes'][keep].cpu().numpy(),
                    'labels': pred['labels'][keep].cpu().numpy(),
                    'scores': pred['scores'][keep].cpu().numpy()
                }
            else:
                filtered_pred = {
                    'boxes': np.empty((0, 4)),
                    'labels': np.empty((0,), dtype=int),
                    'scores': np.empty((0,))
                }
            
            # Ground truth
            gt = {
                'boxes': target['boxes'].cpu().numpy(),
                'labels': target['labels'].cpu().numpy()
            }
            
            # Update ground truth counts
            for iou_thresh in self.iou_thresholds:
                for label in gt['labels']:
                    self.results[iou_thresh]['gt_counts'][label] += 1
            
            # Process each IoU threshold
            for iou_thresh in self.iou_thresholds:
                self._match_predictions_streaming(filtered_pred, gt, iou_thresh)
    
    def _match_predictions_streaming(self, predictions, ground_truths, iou_threshold):
        """Match predictions to ground truth for streaming evaluation"""
        
        if len(predictions['boxes']) == 0:
            return
        
        if len(ground_truths['boxes']) == 0:
            # All predictions are false positives
            for pred_class, pred_score in zip(predictions['labels'], predictions['scores']):
                self.results[iou_threshold]['fp_scores'][pred_class].append(pred_score)
            return
        
        # Sort predictions by confidence
        pred_indices = np.argsort(predictions['scores'])[::-1]
        used_gt = set()
        
        for pred_idx in pred_indices:
            pred_box = predictions['boxes'][pred_idx]
            pred_class = predictions['labels'][pred_idx]
            pred_score = predictions['scores'][pred_idx]
            
            best_iou = 0
            best_gt_idx = -1
            
            # Find best matching ground truth
            for gt_idx, (gt_box, gt_class) in enumerate(zip(ground_truths['boxes'], ground_truths['labels'])):
                if gt_idx in used_gt or gt_class != pred_class:
                    continue
                
                iou = calculate_iou_fast(pred_box, gt_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if match is above threshold
            if best_iou >= iou_threshold and best_gt_idx != -1:
                self.results[iou_threshold]['tp_scores'][pred_class].append(pred_score)
                used_gt.add(best_gt_idx)
            else:
                self.results[iou_threshold]['fp_scores'][pred_class].append(pred_score)
    
    def calculate_ap_per_class(self, tp_scores, fp_scores, num_gt):
        """Calculate AP for a single class using efficient method"""
        if num_gt == 0:
            return 0.0 if len(fp_scores) > 0 else 1.0
        
        if len(tp_scores) == 0 and len(fp_scores) == 0:
            return 0.0
        
        # Combine and sort by score
        combined = [(score, 1) for score in tp_scores] + [(score, 0) for score in fp_scores]
        combined.sort(key=lambda x: x[0], reverse=True)
        
        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls = []
        
        for score, is_tp in combined:
            if is_tp:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            recall = tp_cumsum / num_gt
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using all-point interpolation
        if len(recalls) == 0:
            return 0.0
        
        recalls = np.array([0] + recalls)
        precisions = np.array([1] + precisions)
        
        # Ensure precision is monotonically decreasing
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        # Calculate area under curve
        indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
        ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices - 1])
        
        return ap
    
    def get_final_results(self):
        """Calculate final AP scores"""
        final_results = {}
        
        for iou_thresh in self.iou_thresholds:
            class_aps = {}
            
            for class_id in range(1, self.num_classes):  # Skip background
                tp_scores = self.results[iou_thresh]['tp_scores'][class_id]
                fp_scores = self.results[iou_thresh]['fp_scores'][class_id]
                num_gt = self.results[iou_thresh]['gt_counts'][class_id]
                
                ap = self.calculate_ap_per_class(tp_scores, fp_scores, num_gt)
                class_aps[class_id] = ap
            
            # Calculate mean AP
            valid_aps = [ap for ap in class_aps.values() if not np.isnan(ap)]
            mean_ap = np.mean(valid_aps) if valid_aps else 0.0
            
            final_results[f'AP@{iou_thresh}'] = {
                'mAP': mean_ap,
                'class_aps': class_aps,
                'total_detections': sum(len(scores) for scores in self.results[iou_thresh]['tp_scores'].values()) + 
                                  sum(len(scores) for scores in self.results[iou_thresh]['fp_scores'].values()),
                'true_positives': sum(len(scores) for scores in self.results[iou_thresh]['tp_scores'].values())
            }
        
        return final_results


def evaluate_model_efficient(model_path, test_dataset, device='cuda', 
                           confidence_threshold=0.5, nms_threshold=0.5,
                           batch_size=2, iou_thresholds=[0.5, 0.75]):
    """Memory-efficient model evaluation"""
    
    print(f"Loading model from: {model_path}")
    
    # Load model
    model = get_model(num_classes=7, dropout_rate=0.0)  # 7 classes for apple/cherry
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Create data loader with small batch size
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False
    )
    
    print(f"Evaluating on {len(test_dataset)} original images...")
    print(f"Batch size: {batch_size}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"NMS threshold: {nms_threshold}")
    print(f"IoU thresholds: {iou_thresholds}")
    
    # Initialize streaming evaluator
    evaluator = StreamingEvaluator(
        num_classes=7,
        class_names=test_dataset.class_names,
        iou_thresholds=iou_thresholds
    )
    
    # Process batches with memory management
    total_processed = 0
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader, desc="Processing batches")):
            if images is None or len(images) == 0:
                continue
            
            try:
                # Move to device
                images = list(img.to(device) for img in images)
                
                # Get predictions
                predictions = model(images)
                
                # Process batch
                evaluator.process_batch(
                    predictions, targets, 
                    confidence_threshold=confidence_threshold,
                    nms_threshold=nms_threshold
                )
                
                total_processed += len(images)
                
                # Clear GPU memory periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
                
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    print(f"Processed {total_processed} images successfully")
    
    # Get final results
    results = evaluator.get_final_results()
    
    # Calculate COCO-style mAP (simplified)
    print("Calculating COCO-style mAP...")
    coco_aps = []
    
    for iou_thresh in np.arange(0.5, 1.0, 0.05):
        # Quick single-threshold evaluation
        temp_evaluator = StreamingEvaluator(7, test_dataset.class_names, [iou_thresh])
        
        # Re-process subset for COCO calculation (use fewer samples)
        subset_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        with torch.no_grad():
            processed_count = 0
            for images, targets in subset_loader:
                if images is None or processed_count >= 50:  # Limit for speed
                    break
                
                try:
                    images = list(img.to(device) for img in images)
                    predictions = model(images)
                    temp_evaluator.process_batch(
                        predictions, targets, 
                        confidence_threshold, nms_threshold
                    )
                    processed_count += len(images)
                except:
                    continue
        
        temp_results = temp_evaluator.get_final_results()
        if f'AP@{iou_thresh:.2f}' in temp_results:
            coco_aps.append(temp_results[f'AP@{iou_thresh:.2f}']['mAP'])
    
    coco_map = np.mean(coco_aps) if coco_aps else 0.0
    results['COCO_mAP'] = coco_map
    
    return results


def create_evaluation_report(results, class_names, save_path=None):
    """Create evaluation report with visualizations"""
    
    print("\n" + "="*70)
    print("APPLE & CHERRY MODEL EVALUATION REPORT")
    print("="*70)
    
    # Print overall results
    print("\nOverall Performance:")
    for metric, data in results.items():
        if isinstance(data, dict) and 'mAP' in data:
            print(f"  {metric}: {data['mAP']:.4f}")
            if 'total_detections' in data:
                precision = data['true_positives'] / data['total_detections'] if data['total_detections'] > 0 else 0
                print(f"    Precision: {precision:.4f}")
                print(f"    Total detections: {data['total_detections']}")
                print(f"    True positives: {data['true_positives']}")
        elif isinstance(data, (int, float)):
            print(f"  {metric}: {data:.4f}")
    
    # Per-class results
    print("\nPer-Class Average Precision (AP@0.5):")
    if 'AP@0.5' in results and 'class_aps' in results['AP@0.5']:
        class_aps = results['AP@0.5']['class_aps']
        
        for class_id, ap in class_aps.items():
            if class_id < len(class_names):
                print(f"  {class_names[class_id]}: {ap:.4f}")
        
        # Group by fruit type
        print("\nAverage AP by fruit type:")
        fruit_aps = defaultdict(list)
        for class_id, ap in class_aps.items():
            if class_id < len(class_names):
                fruit_type = class_names[class_id].split('-')[0]
                fruit_aps[fruit_type].append(ap)
        
        for fruit_type, aps in fruit_aps.items():
            avg_ap = np.mean(aps)
            print(f"  {fruit_type}: {avg_ap:.4f}")
    
    # Create visualizations
    if len(results) > 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Apple & Cherry Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Overall mAP comparison
        ax1 = axes[0, 0]
        metrics = []
        values = []
        for metric, data in results.items():
            if isinstance(data, dict) and 'mAP' in data:
                metrics.append(metric)
                values.append(data['mAP'])
            elif isinstance(data, (int, float)):
                metrics.append(metric)
                values.append(data)
        
        if metrics:
            bars = ax1.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'][:len(metrics)])
            ax1.set_title('Overall mAP Scores')
            ax1.set_ylabel('mAP Score')
            ax1.set_ylim([0, 1])
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 2. Per-class AP heatmap
        ax2 = axes[0, 1]
        if 'AP@0.5' in results and 'class_aps' in results['AP@0.5']:
            class_aps = results['AP@0.5']['class_aps']
            class_names_used = [class_names[cid] for cid in sorted(class_aps.keys()) if cid < len(class_names)]
            ap_values = [class_aps[cid] for cid in sorted(class_aps.keys())]
            
            # Create heatmap-style visualization
            y_pos = np.arange(len(class_names_used))
            bars = ax2.barh(y_pos, ap_values, color=plt.cm.RdYlGn([v for v in ap_values]))
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(class_names_used)
            ax2.set_xlabel('Average Precision')
            ax2.set_title('Per-Class AP@0.5')
            ax2.set_xlim([0, 1])
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, ap_values)):
                width = bar.get_width()
                ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                        f'{value:.3f}', ha='left', va='center')
        
        # 3. Fruit type comparison
        ax3 = axes[1, 0]
        if 'AP@0.5' in results and 'class_aps' in results['AP@0.5']:
            fruit_aps = defaultdict(list)
            for class_id, ap in results['AP@0.5']['class_aps'].items():
                if class_id < len(class_names):
                    fruit_type = class_names[class_id].split('-')[0]
                    fruit_aps[fruit_type].append(ap)
            
            fruits = list(fruit_aps.keys())
            avg_aps = [np.mean(aps) for aps in fruit_aps.values()]
            
            bars = ax3.bar(fruits, avg_aps, color=['red', 'darkred'])
            ax3.set_title('Average AP by Fruit Type')
            ax3.set_ylabel('Average AP')
            ax3.set_ylim([0, 1])
            ax3.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, avg_aps):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Detection statistics
        ax4 = axes[1, 1]
        if 'AP@0.5' in results:
            data = results['AP@0.5']
            total_det = data.get('total_detections', 0)
            true_pos = data.get('true_positives', 0)
            false_pos = total_det - true_pos
            
            categories = ['True Positives', 'False Positives']
            values = [true_pos, false_pos]
            colors = ['green', 'red']
            
            wedges, texts, autotexts = ax4.pie(values, labels=categories, colors=colors, autopct='%1.1f%%')
            ax4.set_title('Detection Results Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plot_path = f"{save_path}_evaluation_report.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"\nEvaluation plot saved to: {plot_path}")
        
        plt.show()
    
    # Save summary to CSV
    if save_path:
        summary_data = []
        for metric, data in results.items():
            if isinstance(data, dict) and 'mAP' in data:
                summary_data.append({
                    'Metric': metric,
                    'mAP': data['mAP'],
                    'Total_Detections': data.get('total_detections', 0),
                    'True_Positives': data.get('true_positives', 0)
                })
            elif isinstance(data, (int, float)):
                summary_data.append({
                    'Metric': metric,
                    'mAP': data,
                    'Total_Detections': 0,
                    'True_Positives': 0
                })
        
        summary_df = pd.DataFrame(summary_data)
        csv_path = f"{save_path}_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"Summary saved to: {csv_path}")
    
    return results


def main_evaluation():
    """Main function for evaluating the apple/cherry model"""
    
    config = {
        'images_dir': 'AugmentedDataset/images/',
        'masks_dir': 'AugmentedDataset/',
        'model_path': 'apple_cherry_model_enhanced/best_model.pth',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 2,  # Small batch size to avoid memory issues
        'confidence_threshold': 0.5,
        'nms_threshold': 0.5,
        'iou_thresholds': [0.5, 0.75],
        'target_fruits': ['apple', 'cherry']
    }
    
    print("Apple & Cherry Model Evaluation")
    print("="*50)
    print(f"Device: {config['device']}")
    print(f"Model: {config['model_path']}")
    print(f"Target fruits: {config['target_fruits']}")
    
    # Check if model exists
    if not os.path.exists(config['model_path']):
        print(f"Error: Model not found at {config['model_path']}")
        # Try alternative paths
        alt_paths = [
            'apple_cherry_model_enhanced/final_model.pth',
            'apple_cherry_model_enhanced/early_stop_best.pth'
        ]
        
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                config['model_path'] = alt_path
                print(f"Using alternative model: {alt_path}")
                break
        else:
            print("No model found. Please check the model path.")
            return
    
    # Create test dataset with only original images
    print("\nLoading original images dataset...")
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = OriginalImagesDataset(
        images_dir=config['images_dir'],
        masks_dir=config['masks_dir'],
        transform=transform_test,
        target_fruits=config['target_fruits']
    )
    
    if len(test_dataset) == 0:
        print("No original images found in dataset!")
        return
    
    print(f"Found {len(test_dataset)} original images for evaluation")
    
    # Run evaluation
    print("\nStarting evaluation...")
    
    try:
        results = evaluate_model_efficient(
            model_path=config['model_path'],
            test_dataset=test_dataset,
            device=config['device'],
            confidence_threshold=config['confidence_threshold'],
            nms_threshold=config['nms_threshold'],
            batch_size=config['batch_size'],
            iou_thresholds=config['iou_thresholds']
        )
        
        # Create evaluation report
        create_evaluation_report(
            results=results,
            class_names=test_dataset.class_names,
            save_path='apple_cherry_evaluation_results'
        )
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        return results
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def quick_test_single_image(model_path, image_path, device='cuda', 
                           confidence_threshold=0.5, nms_threshold=0.5):
    """Quick test on a single image to verify model is working"""
    
    print(f"Testing model on single image: {image_path}")
    
    # Load model
    model = get_model(num_classes=7, dropout_rate=0.0)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_image = image.copy()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(Image.fromarray(image)).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model(image_tensor)
    
    prediction = predictions[0]
    
    # Apply NMS
    prediction = apply_nms_efficient(prediction, nms_threshold)
    
    # Filter by confidence
    if len(prediction['scores']) > 0:
        keep = prediction['scores'] > confidence_threshold
        boxes = prediction['boxes'][keep].cpu().numpy()
        labels = prediction['labels'][keep].cpu().numpy()
        scores = prediction['scores'][keep].cpu().numpy()
    else:
        boxes = np.array([])
        labels = np.array([])
        scores = np.array([])
    
    print(f"Found {len(boxes)} objects with confidence > {confidence_threshold}")
    
    # Visualize if detections found
    if len(boxes) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Predictions
        axes[1].imshow(original_image)
        
        class_names = ['background',
                      'apple-ripe', 'apple-unripe', 'apple-spoiled',
                      'cherry-ripe', 'cherry-unripe', 'cherry-spoiled']
        
        colors = {'apple': 'red', 'cherry': 'darkred'}
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            class_name = class_names[label]
            fruit_type = class_name.split('-')[0]
            color = colors.get(fruit_type, 'blue')
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor=color, linewidth=2)
            axes[1].add_patch(rect)
            
            # Add label
            axes[1].text(x1, y1-5, f"{class_name}: {score:.2f}", 
                        bbox=dict(facecolor=color, alpha=0.7),
                        fontsize=10, color='white', weight='bold')
            
            print(f"  - {class_name}: {score:.3f}")
        
        axes[1].set_title(f'Detections (conf>{confidence_threshold})')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return len(boxes)


def find_sample_images(images_dir, target_fruits=['apple', 'cherry'], max_samples=5):
    """Find sample original images for testing"""
    sample_images = []
    
    for fruit in target_fruits:
        fruit_dir = os.path.join(images_dir, fruit)
        if not os.path.exists(fruit_dir):
            continue
        
        for filename in os.listdir(fruit_dir):
            if '_original' in filename and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample_images.append(os.path.join(fruit_dir, filename))
                if len(sample_images) >= max_samples:
                    break
        
        if len(sample_images) >= max_samples:
            break
    
    return sample_images


if __name__ == "__main__":
    # Quick test first
    print("Apple & Cherry Model Evaluation Script")
    print("="*50)
    
    # Check if we should do a quick test first
    quick_test = input("Do you want to run a quick test on a sample image first? (y/n): ").lower().strip()
    
    if quick_test == 'y':
        # Find sample images
        sample_images = find_sample_images('AugmentedDataset/images/')
        
        if sample_images:
            print(f"\nFound {len(sample_images)} sample images")
            for i, img_path in enumerate(sample_images[:3]):
                print(f"{i+1}. {os.path.basename(img_path)}")
            
            try:
                choice = int(input(f"Choose image (1-{min(3, len(sample_images))}): ")) - 1
                if 0 <= choice < len(sample_images):
                    model_path = 'apple_cherry_model_enhanced/best_model.pth'
                    if not os.path.exists(model_path):
                        model_path = 'apple_cherry_model_enhanced/final_model.pth'
                    
                    if os.path.exists(model_path):
                        detections = quick_test_single_image(
                            model_path=model_path,
                            image_path=sample_images[choice],
                            device='cuda' if torch.cuda.is_available() else 'cpu'
                        )
                        
                        if detections is not None:
                            print(f"Quick test completed - found {detections} detections")
                        else:
                            print("Quick test failed")
                    else:
                        print("No model found for quick test")
                else:
                    print("Invalid choice")
            except:
                print("Invalid input")
        else:
            print("No sample images found")
    
    # Run full evaluation
    print("\nRunning full evaluation...")
    run_full = input("Proceed with full evaluation? (y/n): ").lower().strip()
    
    if run_full == 'y':
        results = main_evaluation()
        
        if results:
            print("\nEvaluation completed successfully!")
            print("Check the generated reports and plots for detailed results.")
        else:
            print("Evaluation failed - check error messages above.")
    else:
        print("Evaluation cancelled.")