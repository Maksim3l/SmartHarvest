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
warnings.filterwarnings('ignore')

# Import your existing dataset and model classes
# Note: You'll need to ensure these are available from your Training module
from Training import FruitInstanceDataset, get_model, FastRCNNPredictorWithDropout, collate_fn

# Define which classes to evaluate (apples and cherries only)
APPLE_CHERRY_CLASSES = {
    1: 'apple-ripe',
    2: 'apple-unripe', 
    3: 'apple-spoiled',
    4: 'cherry-ripe',
    5: 'cherry-unripe',
    6: 'cherry-spoiled'
}

# Full class names for reference (but we'll only use apple and cherry)
FULL_CLASS_NAMES = ['background',
                    'apple-ripe', 'apple-unripe', 'apple-spoiled',
                    'cherry-ripe', 'cherry-unripe', 'cherry-spoiled',
                    'cucumber-ripe', 'cucumber-unripe', 'cucumber-spoiled',
                    'strawberry-ripe', 'strawberry-unripe', 'strawberry-spoiled',
                    'tomato-ripe', 'tomato-unripe', 'tomato-spoiled']


def filter_apple_cherry_only(data_dict, is_numpy=False):
    """
    Filter predictions or ground truth to only include apple and cherry classes
    """
    if is_numpy:
        # For numpy arrays
        valid_indices = []
        for idx, label in enumerate(data_dict['labels']):
            if label in APPLE_CHERRY_CLASSES.keys():
                valid_indices.append(idx)
        
        if len(valid_indices) == 0:
            return {
                'boxes': np.array([]),
                'labels': np.array([]),
                'scores': np.array([]) if 'scores' in data_dict else None,
                'masks': np.array([]) if 'masks' in data_dict else None
            }
        
        filtered = {
            'boxes': data_dict['boxes'][valid_indices],
            'labels': data_dict['labels'][valid_indices]
        }
        
        if 'scores' in data_dict:
            filtered['scores'] = data_dict['scores'][valid_indices]
        if 'masks' in data_dict and data_dict['masks'] is not None:
            filtered['masks'] = data_dict['masks'][valid_indices]
            
    else:
        # For torch tensors
        valid_mask = torch.zeros(len(data_dict['labels']), dtype=torch.bool)
        for class_id in APPLE_CHERRY_CLASSES.keys():
            valid_mask |= (data_dict['labels'] == class_id)
        
        filtered = {
            'boxes': data_dict['boxes'][valid_mask],
            'labels': data_dict['labels'][valid_mask]
        }
        
        if 'scores' in data_dict:
            filtered['scores'] = data_dict['scores'][valid_mask]
        if 'masks' in data_dict and data_dict['masks'] is not None:
            filtered['masks'] = data_dict['masks'][valid_mask]
    
    return filtered


def apply_nms(prediction, nms_threshold=0.5):
    """
    Apply Non-Maximum Suppression to predictions
    """
    if len(prediction['boxes']) == 0:
        return prediction
    
    # Filter to apple and cherry classes first
    prediction = filter_apple_cherry_only(prediction, is_numpy=False)
    
    if len(prediction['boxes']) == 0:
        return prediction
    
    # Group boxes by class
    keep_masks = []
    
    for class_id in prediction['labels'].unique():
        # Only process apple and cherry classes
        if class_id.item() not in APPLE_CHERRY_CLASSES.keys():
            continue
            
        # Get indices for this class
        class_mask = prediction['labels'] == class_id
        class_indices = torch.where(class_mask)[0]
        
        if len(class_indices) > 0:
            # Apply NMS for this class
            class_boxes = prediction['boxes'][class_indices]
            class_scores = prediction['scores'][class_indices]
            
            # NMS returns indices to keep
            keep_indices = nms(class_boxes, class_scores, nms_threshold)
            
            # Map back to original indices
            keep_masks.append(class_indices[keep_indices])
    
    # Combine all kept indices
    if keep_masks:
        keep = torch.cat(keep_masks)
        
        # Sort by score for consistent ordering
        keep = keep[prediction['scores'][keep].argsort(descending=True)]
        
        # Filter predictions
        filtered_pred = {
            'boxes': prediction['boxes'][keep],
            'labels': prediction['labels'][keep],
            'scores': prediction['scores'][keep]
        }
        
        if 'masks' in prediction and prediction['masks'] is not None:
            filtered_pred['masks'] = prediction['masks'][keep]
        
        return filtered_pred
    else:
        # Return empty predictions
        return {
            'boxes': prediction['boxes'][:0],
            'labels': prediction['labels'][:0],
            'scores': prediction['scores'][:0],
            'masks': prediction['masks'][:0] if 'masks' in prediction else None
        }


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes
    box format: [x1, y1, x2, y2]
    """
    # Calculate intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Calculate intersection area
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        intersection = 0
    else:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_mask_iou(mask1, mask2):
    """Calculate IoU for segmentation masks"""
    # Ensure masks are binary and 2D
    if len(mask1.shape) > 2:
        mask1 = mask1.squeeze()
    if len(mask2.shape) > 2:
        mask2 = mask2.squeeze()
    
    # Convert to binary
    mask1 = mask1 > 0.5
    mask2 = mask2 > 0.5
    
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0


def match_predictions_to_gt(predictions, ground_truths, iou_threshold=0.5, use_masks=False):
    """
    Match predictions to ground truth boxes/masks based on IoU
    Only considers apple and cherry classes
    """
    matches = []
    used_gt = set()
    unmatched_preds = []
    
    # Filter to apple and cherry only
    predictions = filter_apple_cherry_only(predictions, is_numpy=True)
    ground_truths = filter_apple_cherry_only(ground_truths, is_numpy=True)
    
    # Handle empty predictions
    if len(predictions['boxes']) == 0 or predictions['boxes'].size == 0:
        unmatched_gts = [{'gt_idx': i, 'class': ground_truths['labels'][i]} 
                        for i in range(len(ground_truths['labels']))]
        return matches, unmatched_preds, unmatched_gts
    
    # Handle empty ground truths
    if len(ground_truths['boxes']) == 0 or ground_truths['boxes'].size == 0:
        for pred_idx in range(len(predictions['scores'])):
            unmatched_preds.append({
                'pred_idx': pred_idx,
                'class': predictions['labels'][pred_idx],
                'score': predictions['scores'][pred_idx],
                'best_iou': 0
            })
        return matches, unmatched_preds, []
    
    # Sort predictions by confidence score (descending)
    pred_indices = sorted(range(len(predictions['scores'])), 
                         key=lambda i: predictions['scores'][i], reverse=True)
    
    for pred_idx in pred_indices:
        pred_box = predictions['boxes'][pred_idx]
        pred_class = predictions['labels'][pred_idx]
        pred_score = predictions['scores'][pred_idx]
        
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth of same class
        for gt_idx, (gt_box, gt_class) in enumerate(zip(ground_truths['boxes'], ground_truths['labels'])):
            if gt_idx in used_gt or gt_class != pred_class:
                continue
            
            if use_masks and 'masks' in predictions and 'masks' in ground_truths:
                if predictions['masks'] is not None and ground_truths['masks'] is not None:
                    pred_mask = predictions['masks'][pred_idx]
                    gt_mask = ground_truths['masks'][gt_idx]
                    iou = calculate_mask_iou(pred_mask, gt_mask)
                else:
                    iou = calculate_iou(pred_box, gt_box)
            else:
                iou = calculate_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is above threshold
        if best_iou >= iou_threshold and best_gt_idx != -1:
            matches.append({
                'pred_idx': pred_idx,
                'gt_idx': best_gt_idx,
                'iou': best_iou,
                'class': pred_class,
                'score': pred_score
            })
            used_gt.add(best_gt_idx)
        else:
            unmatched_preds.append({
                'pred_idx': pred_idx,
                'class': pred_class,
                'score': pred_score,
                'best_iou': best_iou
            })
    
    # Unmatched ground truths
    unmatched_gts = [{'gt_idx': i, 'class': ground_truths['labels'][i]} 
                     for i in range(len(ground_truths['labels'])) if i not in used_gt]
    
    return matches, unmatched_preds, unmatched_gts


def calculate_ap_for_class(tp_scores, fp_scores, num_gt, interpolation='11point'):
    """
    Calculate Average Precision for a single class
    """
    if num_gt == 0:
        return 0.0 if len(fp_scores) > 0 else 1.0
    
    if len(tp_scores) == 0 and len(fp_scores) == 0:
        return 0.0
    
    # Combine and sort by score
    combined = [(score, 1) for score in tp_scores] + [(score, 0) for score in fp_scores]
    combined.sort(key=lambda x: x[0], reverse=True)
    
    # Calculate precision and recall at each threshold
    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []
    
    for score, is_tp in combined:
        if is_tp:
            tp_cumsum += 1
        else:
            fp_cumsum += 1
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum) > 0 else 0
        recall = tp_cumsum / num_gt if num_gt > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    
    # Calculate AP
    if interpolation == '11point':
        # 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            # Find precision at recall >= t
            precisions_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            max_precision = max(precisions_at_recall) if precisions_at_recall else 0
            ap += max_precision
        ap /= 11
    else:
        # All-point interpolation (COCO style)
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


def evaluate_single_model(model, test_loader, device, confidence_threshold=0.5,
                         nms_threshold=0.5, iou_thresholds=[0.5, 0.75], 
                         use_masks=False):
    """
    Evaluate a single model on test data with NMS
    Only evaluates apple and cherry classes
    """
    model.eval()
    
    # Collect all predictions and ground truths
    all_matches = {iou_thresh: [] for iou_thresh in iou_thresholds}
    all_unmatched_preds = {iou_thresh: [] for iou_thresh in iou_thresholds}
    class_gt_counts = defaultdict(int)
    
    print("Collecting predictions and ground truths (Apples and Cherries only)...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader)):
            if images is None or len(images) == 0:
                continue
            
            # Clear cache periodically to manage memory
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                # Move to device
                images = list(img.to(device) for img in images)
                
                # Get predictions
                predictions = model(images)
                
                # Process each image in batch
                for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
                    # Apply NMS FIRST (before confidence filtering)
                    pred = apply_nms(pred, nms_threshold)
                    
                    # Then filter by confidence
                    if len(pred['scores']) > 0:
                        keep = pred['scores'] > confidence_threshold
                    else:
                        keep = torch.zeros(0, dtype=torch.bool)
                    
                    filtered_pred = {
                        'boxes': pred['boxes'][keep].cpu().numpy() if keep.sum() > 0 else np.array([]),
                        'labels': pred['labels'][keep].cpu().numpy() if keep.sum() > 0 else np.array([]),
                        'scores': pred['scores'][keep].cpu().numpy() if keep.sum() > 0 else np.array([])
                    }
                    
                    if use_masks and 'masks' in pred and pred['masks'] is not None:
                        filtered_pred['masks'] = pred['masks'][keep].cpu().numpy() if keep.sum() > 0 else np.array([])
                    
                    # Prepare ground truth and filter for apple/cherry
                    gt = {
                        'boxes': target['boxes'].cpu().numpy(),
                        'labels': target['labels'].cpu().numpy()
                    }
                    
                    if use_masks and 'masks' in target:
                        gt['masks'] = target['masks'].cpu().numpy()
                    
                    # Filter ground truth to apple and cherry only
                    gt = filter_apple_cherry_only(gt, is_numpy=True)
                    
                    # Count ground truth instances per class (apple and cherry only)
                    for label in gt['labels']:
                        if label in APPLE_CHERRY_CLASSES.keys():
                            class_gt_counts[label] += 1
                    
                    # Match predictions for each IoU threshold
                    for iou_thresh in iou_thresholds:
                        matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(
                            filtered_pred, gt, iou_thresh, use_masks
                        )
                        
                        all_matches[iou_thresh].extend(matches)
                        all_unmatched_preds[iou_thresh].extend(unmatched_preds)
                        
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue
    
    # Calculate AP for each class and IoU threshold
    results = {}
    
    for iou_thresh in iou_thresholds:
        print(f"\nCalculating AP@{iou_thresh} for Apple and Cherry classes...")
        
        # Group matches and false positives by class
        class_matches = defaultdict(list)
        class_false_positives = defaultdict(list)
        
        for match in all_matches[iou_thresh]:
            if match['class'] in APPLE_CHERRY_CLASSES.keys():
                class_matches[match['class']].append(match['score'])
        
        for fp in all_unmatched_preds[iou_thresh]:
            if fp['class'] in APPLE_CHERRY_CLASSES.keys():
                class_false_positives[fp['class']].append(fp['score'])
        
        # Calculate AP for each apple and cherry class
        class_aps = {}
        for class_id in APPLE_CHERRY_CLASSES.keys():
            tp_scores = class_matches[class_id]
            fp_scores = class_false_positives[class_id]
            num_gt = class_gt_counts[class_id]
            
            ap = calculate_ap_for_class(tp_scores, fp_scores, num_gt, interpolation='all')
            class_aps[class_id] = ap
        
        # Calculate mean AP (only for apple and cherry)
        valid_aps = [ap for ap in class_aps.values() if not np.isnan(ap)]
        mean_ap = np.mean(valid_aps) if valid_aps else 0.0
        
        results[f'AP@{iou_thresh}'] = {
            'mAP': mean_ap,
            'class_aps': class_aps,
            'num_predictions': len(all_matches[iou_thresh]) + len(all_unmatched_preds[iou_thresh]),
            'num_matches': len(all_matches[iou_thresh])
        }
    
    # Calculate COCO-style mAP (average over IoU 0.5:0.95) for apple and cherry
    print(f"\nCalculating COCO mAP for Apple and Cherry classes...")
    coco_map = calculate_coco_map_simplified(
        model, test_loader, device, confidence_threshold, 
        nms_threshold, use_masks
    )
    results['COCO_mAP'] = coco_map
    
    return results


def calculate_coco_map_simplified(model, test_loader, device, confidence_threshold=0.5,
                                  nms_threshold=0.5, use_masks=False):
    """
    Simplified COCO mAP calculation for efficiency
    Only for apple and cherry classes
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    all_aps = []
    
    for iou_thresh in tqdm(iou_thresholds, desc="COCO mAP IoU thresholds (Apple & Cherry)"):
        # Run evaluation for this specific IoU threshold
        results = evaluate_single_iou_threshold(
            model, test_loader, device, confidence_threshold,
            nms_threshold, iou_thresh, use_masks
        )
        all_aps.append(results['mAP'])
    
    return np.mean(all_aps)


def evaluate_single_iou_threshold(model, test_loader, device, confidence_threshold,
                                  nms_threshold, iou_threshold, use_masks):
    """
    Evaluate at a single IoU threshold (helper for COCO mAP)
    Only for apple and cherry classes
    """
    model.eval()
    all_matches = []
    all_unmatched_preds = []
    class_gt_counts = defaultdict(int)
    
    with torch.no_grad():
        for images, targets in test_loader:
            if images is None or len(images) == 0:
                continue
            
            try:
                images = list(img.to(device) for img in images)
                predictions = model(images)
                
                for pred, target in zip(predictions, targets):
                    # Apply NMS and confidence filtering
                    pred = apply_nms(pred, nms_threshold)
                    
                    if len(pred['scores']) > 0:
                        keep = pred['scores'] > confidence_threshold
                    else:
                        keep = torch.zeros(0, dtype=torch.bool)
                    
                    filtered_pred = {
                        'boxes': pred['boxes'][keep].cpu().numpy() if keep.sum() > 0 else np.array([]),
                        'labels': pred['labels'][keep].cpu().numpy() if keep.sum() > 0 else np.array([]),
                        'scores': pred['scores'][keep].cpu().numpy() if keep.sum() > 0 else np.array([])
                    }
                    
                    if use_masks and 'masks' in pred and pred['masks'] is not None:
                        filtered_pred['masks'] = pred['masks'][keep].cpu().numpy() if keep.sum() > 0 else np.array([])
                    
                    gt = {
                        'boxes': target['boxes'].cpu().numpy(),
                        'labels': target['labels'].cpu().numpy()
                    }
                    
                    if use_masks and 'masks' in target:
                        gt['masks'] = target['masks'].cpu().numpy()
                    
                    # Filter to apple and cherry only
                    gt = filter_apple_cherry_only(gt, is_numpy=True)
                    
                    for label in gt['labels']:
                        if label in APPLE_CHERRY_CLASSES.keys():
                            class_gt_counts[label] += 1
                    
                    matches, unmatched_preds, _ = match_predictions_to_gt(
                        filtered_pred, gt, iou_threshold, use_masks
                    )
                    
                    all_matches.extend(matches)
                    all_unmatched_preds.extend(unmatched_preds)
                    
            except Exception:
                continue
    
    # Calculate AP for each apple and cherry class
    class_matches = defaultdict(list)
    class_false_positives = defaultdict(list)
    
    for match in all_matches:
        if match['class'] in APPLE_CHERRY_CLASSES.keys():
            class_matches[match['class']].append(match['score'])
    
    for fp in all_unmatched_preds:
        if fp['class'] in APPLE_CHERRY_CLASSES.keys():
            class_false_positives[fp['class']].append(fp['score'])
    
    class_aps = {}
    for class_id in APPLE_CHERRY_CLASSES.keys():
        tp_scores = class_matches[class_id]
        fp_scores = class_false_positives[class_id]
        num_gt = class_gt_counts[class_id]
        
        ap = calculate_ap_for_class(tp_scores, fp_scores, num_gt, interpolation='all')
        class_aps[class_id] = ap
    
    valid_aps = [ap for ap in class_aps.values() if not np.isnan(ap)]
    mean_ap = np.mean(valid_aps) if valid_aps else 0.0
    
    return {'mAP': mean_ap, 'class_aps': class_aps}


def compare_models(model_paths, test_loader, device, model_names=None, **eval_kwargs):
    """
    Compare multiple models on the same test data
    """
    if model_names is None:
        model_names = [f'Model_{i+1}' for i in range(len(model_paths))]
    
    results = {}
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name} (Apple & Cherry Classes Only)")
        print(f"Model path: {model_path}")
        print(f"{'='*50}")
        
        # Load checkpoint first to determine number of classes
        checkpoint = torch.load(model_path, map_location=device)
        
        # Determine the number of classes from the checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Get number of classes from the classifier layer
        num_classes = state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]
        print(f"Model was trained with {num_classes} classes (including background)")
        
        # Create model with correct number of classes
        model = get_model(num_classes=num_classes, dropout_rate=0.0)  # No dropout for inference
        
        # Load the state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        
        # Evaluate model
        model_results = evaluate_single_model(model, test_loader, device, **eval_kwargs)
        results[model_name] = model_results
        
        # Print summary
        print(f"\n{model_name} Results (Apple & Cherry):")
        for metric, data in model_results.items():
            if isinstance(data, dict) and 'mAP' in data:
                print(f"  {metric}: {data['mAP']:.4f}")
            elif isinstance(data, (int, float)):
                print(f"  {metric}: {data:.4f}")
        
        # Clear GPU memory after each model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def create_comparison_report(results, save_path=None):
    """
    Create a detailed comparison report of model results for apple and cherry only
    """
    # Create summary table
    summary_data = []
    
    for model_name, model_results in results.items():
        row = {'Model': model_name}
        
        # Add main metrics
        for metric, data in model_results.items():
            if isinstance(data, dict) and 'mAP' in data:
                row[metric] = data['mAP']
            elif isinstance(data, (int, float)):
                row[metric] = data
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create detailed per-class comparison
    per_class_data = []
    
    for model_name, model_results in results.items():
        for metric, data in model_results.items():
            if isinstance(data, dict) and 'class_aps' in data:
                for class_id, ap in data['class_aps'].items():
                    if class_id in APPLE_CHERRY_CLASSES:
                        per_class_data.append({
                            'Model': model_name,
                            'Metric': metric,
                            'Class_ID': class_id,
                            'Class_Name': APPLE_CHERRY_CLASSES[class_id],
                            'AP': ap
                        })
    
    per_class_df = pd.DataFrame(per_class_data) if per_class_data else pd.DataFrame()
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # 1. Overall mAP comparison
    if 'AP@0.5' in summary_df.columns:
        ax1 = axes[0, 0]
        metrics_to_plot = [col for col in ['AP@0.5', 'AP@0.75', 'COCO_mAP'] if col in summary_df.columns]
        summary_df.plot(x='Model', y=metrics_to_plot, kind='bar', ax=ax1, rot=45)
        ax1.set_title('Apple & Cherry Model Comparison - Overall mAP', fontsize=14, fontweight='bold')
        ax1.set_ylabel('mAP Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
    
    # 2. Per-class AP@0.5 heatmap
    if not per_class_df.empty:
        ax2 = axes[0, 1]
        # Create pivot table for heatmap
        pivot_data = per_class_df[per_class_df['Metric'] == 'AP@0.5'].pivot(
            index='Class_Name', columns='Model', values='AP'
        )
        
        if not pivot_data.empty:
            im = ax2.imshow(pivot_data.values, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            ax2.set_xticks(range(len(pivot_data.columns)))
            ax2.set_xticklabels(pivot_data.columns, rotation=45)
            ax2.set_yticks(range(len(pivot_data.index)))
            ax2.set_yticklabels(pivot_data.index)
            ax2.set_title('Apple & Cherry Per-Class AP@0.5 Heatmap', fontsize=14, fontweight='bold')
            
            # Add colorbar
            plt.colorbar(im, ax=ax2)
            
            # Add text annotations
            for i in range(len(pivot_data.index)):
                for j in range(len(pivot_data.columns)):
                    value = pivot_data.iloc[i, j]
                    if not np.isnan(value):
                        text = ax2.text(j, i, f'{value:.2f}',
                                       ha="center", va="center", 
                                       color="white" if value < 0.5 else "black", 
                                       fontsize=8)
    else:
        ax2 = axes[0, 1]
        ax2.text(0.5, 0.5, 'No per-class data available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Model performance by ripeness stage (apple and cherry)
    ax3 = axes[1, 0]
    if not per_class_df.empty:
        # Group by ripeness stage
        ripeness_data = []
        for _, row in per_class_df[per_class_df['Metric'] == 'AP@0.5'].iterrows():
            fruit_type = row['Class_Name'].split('-')[0]
            ripeness = row['Class_Name'].split('-')[1]
            ripeness_data.append({
                'Model': row['Model'],
                'Fruit': fruit_type,
                'Ripeness': ripeness,
                'AP': row['AP']
            })
        
        if ripeness_data:
            ripeness_df = pd.DataFrame(ripeness_data)
            # Average AP by ripeness stage
            ripeness_avg = ripeness_df.groupby(['Model', 'Ripeness'])['AP'].mean().reset_index()
            ripeness_pivot = ripeness_avg.pivot(index='Ripeness', columns='Model', values='AP')
            ripeness_pivot.plot(kind='bar', ax=ax3, rot=45)
            ax3.set_title('Apple & Cherry AP@0.5 by Ripeness Stage', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Average AP')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1])
    
    # 4. Apple vs Cherry comparison
    ax4 = axes[1, 1]
    if not per_class_df.empty:
        fruit_comparison = []
        for _, row in per_class_df[per_class_df['Metric'] == 'AP@0.5'].iterrows():
            fruit_type = row['Class_Name'].split('-')[0]
            fruit_comparison.append({
                'Model': row['Model'],
                'Fruit': fruit_type,
                'AP': row['AP']
            })
        
        if fruit_comparison:
            fruit_comp_df = pd.DataFrame(fruit_comparison)
            fruit_avg = fruit_comp_df.groupby(['Model', 'Fruit'])['AP'].mean().reset_index()
            fruit_pivot = fruit_avg.pivot(index='Fruit', columns='Model', values='AP')
            fruit_pivot.plot(kind='bar', ax=ax4, rot=0)
            ax4.set_title('Apple vs Cherry Average AP@0.5', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Average AP')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save results
    if save_path:
        # Save plots
        plt.savefig(f"{save_path}_apple_cherry_comparison.png", dpi=300, bbox_inches='tight')
        
        # Save tables
        summary_df.to_csv(f"{save_path}_apple_cherry_summary.csv", index=False)
        if not per_class_df.empty:
            per_class_df.to_csv(f"{save_path}_apple_cherry_per_class.csv", index=False)
        
        print(f"\nResults saved to:")
        print(f"  - {save_path}_apple_cherry_comparison.png")
        print(f"  - {save_path}_apple_cherry_summary.csv")
        if not per_class_df.empty:
            print(f"  - {save_path}_apple_cherry_per_class.csv")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("APPLE & CHERRY MODEL COMPARISON SUMMARY")
    print("="*70)
    print("\nOverall Performance (Apple & Cherry Only):")
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    if not per_class_df.empty:
        print("\nBest performing model per metric:")
        for metric in ['AP@0.5', 'AP@0.75', 'COCO_mAP']:
            if metric in summary_df.columns:
                best_model = summary_df.loc[summary_df[metric].idxmax(), 'Model']
                best_score = summary_df[metric].max()
                print(f"  {metric}: {best_model} ({best_score:.4f})")
        
        # Print per-class best performers
        print("\nPer-Class Performance (AP@0.5):")
        if 'AP@0.5' in per_class_df['Metric'].values:
            ap05_df = per_class_df[per_class_df['Metric'] == 'AP@0.5']
            for class_name in APPLE_CHERRY_CLASSES.values():
                class_data = ap05_df[ap05_df['Class_Name'] == class_name]
                if not class_data.empty:
                    best_ap = class_data['AP'].max()
                    print(f"  {class_name}: {best_ap:.4f}")
    
    return summary_df, per_class_df


def visualize_predictions(model, image_path, device, confidence_threshold=0.5, 
                         nms_threshold=0.5):
    """
    Visualize predictions on a single image (apple and cherry only)
    """
    # Define colors for apple and cherry
    fruit_colors = {
        'apple': (255, 0, 0),      # Red
        'cherry': (139, 0, 0),     # Dark Red
    }
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    # Apply NMS (which also filters to apple/cherry)
    predictions = apply_nms(predictions, nms_threshold)
    
    # Filter by confidence
    if len(predictions['scores']) > 0:
        keep = predictions['scores'] > confidence_threshold
        boxes = predictions['boxes'][keep].cpu().numpy()
        labels = predictions['labels'][keep].cpu().numpy()
        scores = predictions['scores'][keep].cpu().numpy()
    else:
        boxes = np.array([])
        labels = np.array([])
        scores = np.array([])
    
    # Convert image to numpy for drawing
    img_np = np.array(image)
    img_with_boxes = img_np.copy()
    
    # Draw predictions
    for box, label, score in zip(boxes, labels, scores):
        if label not in APPLE_CHERRY_CLASSES.keys():
            continue
            
        x1, y1, x2, y2 = box.astype(int)
        
        # Get color based on fruit type
        class_name = APPLE_CHERRY_CLASSES[label]
        fruit_type = class_name.split('-')[0]
        color = fruit_colors.get(fruit_type, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with score
        label_text = f"{class_name}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(img_with_boxes, 
                     (x1, y1 - label_size[1] - 4),
                     (x1 + label_size[0], y1),
                     color, -1)
        
        # Draw label text
        cv2.putText(img_with_boxes, label_text,
                   (x1, y1 - 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 1)
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(img_with_boxes)
    axes[1].set_title(f"Apple & Cherry Predictions (Conf > {confidence_threshold}, NMS: {nms_threshold})")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Count detections by type
    apple_count = sum(1 for l in labels if l in [1, 2, 3])
    cherry_count = sum(1 for l in labels if l in [4, 5, 6])
    
    print(f"Found {len(boxes)} objects (Apples: {apple_count}, Cherries: {cherry_count})")
    for label, score in zip(labels, scores):
        if label in APPLE_CHERRY_CLASSES:
            print(f"  - {APPLE_CHERRY_CLASSES[label]}: {score:.3f}")
    
    return boxes, labels, scores


def main_evaluation():
    """
    Main function to evaluate models on apple and cherry classes only
    """
    # Configuration
    config = {
        'test_images_dir': 'AugmentedDataset/images/',
        'test_masks_dir': 'AugmentedDataset/',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 4,
        'num_workers': 0,  # Set to 0 to avoid multiprocessing issues
        'confidence_threshold': 0.5,
        'nms_threshold': 0.5,  # NMS threshold
        'iou_thresholds': [0.5, 0.75],
        'use_masks': True  # Set to True if you want to evaluate mask IoU
    }
    
    # Model paths - update these to your actual model paths
    model_paths = [
        'apple_cherry_model_enhanced/best_model.pth'
    ]
    
    model_names = ['Apple_Cherry_Model']
    
    print(f"Device: {config['device']}")
    print(f"Evaluating ONLY Apple and Cherry classes: {list(APPLE_CHERRY_CLASSES.values())}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create test dataset
    print("\nLoading test dataset...")
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Use your existing dataset class
    test_dataset = FruitInstanceDataset(
        images_dir=config['test_images_dir'],
        masks_dir=config['test_masks_dir'],
        transform=transform_test,
        split='test',
        augment=False
    )
    
    # Create data loader with collate_fn from Training module
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,  # Import from Training module
        num_workers=config['num_workers'],
        pin_memory=False
    )
    
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    print(f"Number of batches: {len(test_loader)}")
    print(f"Note: Only Apple and Cherry detections will be evaluated")
    
    # Compare models
    results = compare_models(
        model_paths=model_paths,
        test_loader=test_loader,
        device=config['device'],
        model_names=model_names,
        confidence_threshold=config['confidence_threshold'],
        nms_threshold=config['nms_threshold'],
        iou_thresholds=config['iou_thresholds'],
        use_masks=config['use_masks']
    )
    
    # Create comparison report
    summary_df, per_class_df = create_comparison_report(
        results, 
        save_path='apple_cherry_evaluation_results'
    )
    
    # Additional analysis
    print("\n" + "="*70)
    print("DETAILED APPLE & CHERRY ANALYSIS")
    print("="*70)
    
    # Calculate and display additional metrics
    for model_name, model_results in results.items():
        print(f"\n{model_name} Detailed Metrics:")
        
        if 'AP@0.5' in model_results:
            data = model_results['AP@0.5']
            if data['num_predictions'] > 0:
                precision = data['num_matches'] / data['num_predictions']
                print(f"  Precision@0.5 (Apple & Cherry): {precision:.4f}")
                print(f"  Total Apple & Cherry Detections: {data['num_predictions']}")
                print(f"  True Positives: {data['num_matches']}")
                print(f"  False Positives: {data['num_predictions'] - data['num_matches']}")
            
            # Show per-class breakdown
            print(f"\n  Per-Class AP@0.5:")
            for class_id, class_name in APPLE_CHERRY_CLASSES.items():
                if class_id in data['class_aps']:
                    print(f"    {class_name}: {data['class_aps'][class_id]:.4f}")
    
    return results, summary_df, per_class_df


def evaluate_single_image(model_path, image_path, device='cuda', 
                         confidence_threshold=0.5, nms_threshold=0.5):
    """
    Quick function to evaluate a single model on a single image (apple and cherry only)
    """
    # Load checkpoint first to determine number of classes
    checkpoint = torch.load(model_path, map_location=device)
    
    # Determine the number of classes from the checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Get number of classes from the classifier layer
    num_classes = state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]
    
    # Create model with correct number of classes
    model = get_model(num_classes=num_classes, dropout_rate=0.0)
    
    # Load the state dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Visualize predictions
    boxes, labels, scores = visualize_predictions(
        model, image_path, device, 
        confidence_threshold, nms_threshold
    )
    
    return boxes, labels, scores


if __name__ == "__main__":
    # Run full evaluation
    results, summary_df, per_class_df = main_evaluation()
    
    # Optional: Test on a single image
    # evaluate_single_image(
    #     model_path='apple_cherry_model_enhanced/best_model.pth',
    #     image_path='AugmentedDataset/images/test_image.jpg',
    #     device='cuda' if torch.cuda.is_available() else 'cpu'
    # )