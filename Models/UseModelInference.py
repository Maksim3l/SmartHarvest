import torch
import torchvision.transforms as transforms
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os


class FruitDetector:
    """Easy-to-use fruit detector for 1200x1200 images"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        
        self.class_names = ['background',
                           'apple-ripe', 'apple-unripe', 'apple-spoiled',
                           'cherry-ripe', 'cherry-unripe', 'cherry-spoiled',
                           'cucumber-ripe', 'cucumber-unripe', 'cucumber-spoiled',
                           'strawberry-ripe', 'strawberry-unripe', 'strawberry-spoiled',
                           'tomato-ripe', 'tomato-unripe', 'tomato-spoiled']
        
        self.fruit_colors = {
            'apple': (255, 0, 0),      # Red
            'cherry': (139, 0, 0),     # Dark red
            'cucumber': (0, 255, 0),   # Green
            'strawberry': (255, 20, 147),  # Pink
            'tomato': (255, 69, 0)     # Orange-red
        }
        
        self._load_model()
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def _load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}...")

        self.model = maskrcnn_resnet50_fpn(weights=None)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 16)
        
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 16)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f" Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            if 'best_score' in checkpoint:
                print(f" Best validation score: {checkpoint['best_score']:.4f}")
        else:
            state_dict = checkpoint
            print(" Loaded direct state dict")
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(" Model loaded successfully!")
    
    def detect(self, image_path, score_threshold=0.1):  # Lowered default threshold
        """
        Detect fruits in an image
        
        Args:
            image_path: Path to input image (should be 1200x1200)
            score_threshold: Minimum confidence score for detections
            
        Returns:
            dict: Detection results with boxes, labels, scores, masks
        """

        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path

        h, w = image.shape[:2]
        if h != 1200 or w != 1200:
            print(f"Warning: Image size is {w}x{h}, expected 1200x1200")
            print("Resizing to 1200x1200...")
            image = cv2.resize(image, (1200, 1200))
        
        image_tensor = self.transform(Image.fromarray(image))
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)[0]

        print(f"Raw predictions: {len(predictions['boxes'])} total")
        if len(predictions['scores']) > 0:
            print(f"Score range: {predictions['scores'].min():.3f} - {predictions['scores'].max():.3f}")
            print(f"Top 5 scores: {predictions['scores'][:5].cpu().numpy()}")

        keep = predictions['scores'] > score_threshold
        print(f"Keeping {keep.sum()} predictions above threshold {score_threshold}")

        results = {
            'boxes': predictions['boxes'][keep].cpu().numpy(),
            'labels': predictions['labels'][keep].cpu().numpy(),
            'scores': predictions['scores'][keep].cpu().numpy(),
            'masks': predictions['masks'][keep].cpu().numpy(),
            'image': image
        }
        
        results['fruit_names'] = []
        results['ripeness_states'] = []
        
        for label in results['labels']:
            class_name = self.class_names[label]
            if '-' in class_name:
                fruit, ripeness = class_name.split('-')
                results['fruit_names'].append(fruit)
                results['ripeness_states'].append(ripeness)
            else:
                results['fruit_names'].append('unknown')
                results['ripeness_states'].append('unknown')
        
        return results
    
    def visualize(self, results, save_path=None, show_masks=True, title=None):
        """
        Visualize detection results
        
        Args:
            results: Detection results from detect()
            save_path: Optional path to save visualization
            show_masks: Whether to show segmentation masks
            title: Custom title for the plot
        """
        image = results['image'].copy()

        if len(results['boxes']) == 0:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
            ax.imshow(image)
            ax.set_xlim(0, 1200)
            ax.set_ylim(1200, 0)
            ax.axis('off')
            ax.set_title(title or 'No fruits detected', fontsize=16)
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f" Visualization saved to {save_path}")
            
            plt.show()
            return fig

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        
        for i in range(len(results['boxes'])):
            box = results['boxes'][i]
            label = results['labels'][i]
            score = results['scores'][i]
            fruit = results['fruit_names'][i]
            ripeness = results['ripeness_states'][i]
            color = self.fruit_colors.get(fruit, (255, 255, 255))
            color_normalized = [c/255.0 for c in color]
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, edgecolor=color_normalized, 
                               linewidth=3)
            ax.add_patch(rect)

            if show_masks and i < len(results['masks']):
                mask = results['masks'][i, 0] > 0.5
                masked = np.ma.masked_where(~mask, mask)
                ax.imshow(masked, alpha=0.3, cmap='coolwarm')

            label_text = f"{fruit}-{ripeness}: {score:.2f}"
            ax.text(x1, y1-5, label_text,
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=color_normalized, alpha=0.7),
                   fontsize=12, color='white', weight='bold')
        
        ax.set_xlim(0, 1200)
        ax.set_ylim(1200, 0)
        ax.axis('off')
        ax.set_title(title or f'Detected {len(results["boxes"])} fruits', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Visualization saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def compare_models_side_by_side(self, model1_path, model2_path, image_path, score_threshold=0.1):
        """
        Compare predictions from two models side by side
        
        Args:
            model1_path: Path to first model
            model2_path: Path to second model
            image_path: Path to test image
            score_threshold: Minimum confidence score
        """
        print(f"\n{'='*60}")
        print(f"SIDE-BY-SIDE MODEL COMPARISON: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        detector1 = FruitDetector(model1_path)
        detector2 = FruitDetector(model2_path)

        results1 = detector1.detect(image_path, score_threshold)
        results2 = detector2.detect(image_path, score_threshold)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        
        image = results1['image'].copy()
        ax1.imshow(image)
        
        for i in range(len(results1['boxes'])):
            box = results1['boxes'][i]
            score = results1['scores'][i]
            fruit = results1['fruit_names'][i]
            ripeness = results1['ripeness_states'][i]
            
            color = detector1.fruit_colors.get(fruit, (255, 255, 255))
            color_normalized = [c/255.0 for c in color]
            
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, edgecolor=color_normalized, linewidth=3)
            ax1.add_patch(rect)
            
            if i < len(results1['masks']):
                mask = results1['masks'][i, 0] > 0.5
                masked = np.ma.masked_where(~mask, mask)
                ax1.imshow(masked, alpha=0.3, cmap='coolwarm')
            
            label_text = f"{fruit}-{ripeness}: {score:.2f}"
            ax1.text(x1, y1-5, label_text,
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor=color_normalized, alpha=0.7),
                    fontsize=10, color='white', weight='bold')
        
        ax1.set_xlim(0, 1200)
        ax1.set_ylim(1200, 0)
        ax1.axis('off')
        ax1.set_title(f'Model 1: {len(results1["boxes"])} detections\n({os.path.basename(model1_path)})', 
                     fontsize=14, pad=20)
        
        ax2.imshow(image)
        
        for i in range(len(results2['boxes'])):
            box = results2['boxes'][i]
            score = results2['scores'][i]
            fruit = results2['fruit_names'][i]
            ripeness = results2['ripeness_states'][i]
            
            color = detector2.fruit_colors.get(fruit, (255, 255, 255))
            color_normalized = [c/255.0 for c in color]
            
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, edgecolor=color_normalized, linewidth=3)
            ax2.add_patch(rect)

            if i < len(results2['masks']):
                mask = results2['masks'][i, 0] > 0.5
                masked = np.ma.masked_where(~mask, mask)
                ax2.imshow(masked, alpha=0.3, cmap='coolwarm')
            
            label_text = f"{fruit}-{ripeness}: {score:.2f}"
            ax2.text(x1, y1-5, label_text,
                    bbox=dict(boxstyle="round,pad=0.3", 
                             facecolor=color_normalized, alpha=0.7),
                    fontsize=10, color='white', weight='bold')
        
        ax2.set_xlim(0, 1200)
        ax2.set_ylim(1200, 0)
        ax2.axis('off')
        ax2.set_title(f'Model 2: {len(results2["boxes"])} detections\n({os.path.basename(model2_path)})', 
                     fontsize=14, pad=20)
        
        plt.tight_layout()
        
        save_path = f"comparison_{os.path.basename(image_path).split('.')[0]}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Comparison saved to {save_path}")
        
        plt.show()
        
        print(f"\nDETAILED COMPARISON:")
        print(f"{'='*50}")
        print(f"Model 1 ({os.path.basename(model1_path)}):")
        print(f"  Detections: {len(results1['boxes'])}")
        if len(results1['scores']) > 0:
            print(f"  Avg confidence: {np.mean(results1['scores']):.3f}")
            print(f"  Score range: {results1['scores'].min():.3f} - {results1['scores'].max():.3f}")
        
        print(f"\nModel 2 ({os.path.basename(model2_path)}):")
        print(f"  Detections: {len(results2['boxes'])}")
        if len(results2['scores']) > 0:
            print(f"  Avg confidence: {np.mean(results2['scores']):.3f}")
            print(f"  Score range: {results2['scores'].min():.3f} - {results2['scores'].max():.3f}")

        summary1 = detector1.get_summary(results1)
        summary2 = detector2.get_summary(results2)
        
        print(f"\nFruit type comparison:")
        all_fruits = set(summary1['fruits_by_type'].keys()) | set(summary2['fruits_by_type'].keys())
        for fruit in sorted(all_fruits):
            count1 = summary1['fruits_by_type'].get(fruit, {'total': 0})['total']
            count2 = summary2['fruits_by_type'].get(fruit, {'total': 0})['total']
            print(f"  {fruit.capitalize()}: Model1={count1}, Model2={count2}")
        
        return results1, results2
    
    def compare_four_models(self, model_paths, image_path, score_threshold=0.1):
        """
        Compare predictions from four models in a 2x2 grid
        
        Args:
            model_paths: List of 4 model paths
            image_path: Path to test image
            score_threshold: Minimum confidence score
        """
        if len(model_paths) != 4:
            raise ValueError("Exactly 4 model paths required")
        
        print(f"\n{'='*80}")
        print(f"FOUR-MODEL COMPARISON: {os.path.basename(image_path)}")
        print(f"{'='*80}")

        # Load all models and get results
        detectors = []
        all_results = []
        
        for i, model_path in enumerate(model_paths):
            print(f"Loading model {i+1}/4...")
            detector = FruitDetector(model_path)
            detectors.append(detector)
            results = detector.detect(image_path, score_threshold)
            all_results.append(results)

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(24, 24))
        axes = axes.flatten()
        
        for idx, (detector, results, model_path) in enumerate(zip(detectors, all_results, model_paths)):
            ax = axes[idx]
            image = results['image'].copy()
            ax.imshow(image)
            
            # Draw detections
            for i in range(len(results['boxes'])):
                box = results['boxes'][i]
                score = results['scores'][i]
                fruit = results['fruit_names'][i]
                ripeness = results['ripeness_states'][i]
                
                color = detector.fruit_colors.get(fruit, (255, 255, 255))
                color_normalized = [c/255.0 for c in color]
                
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, edgecolor=color_normalized, linewidth=2)
                ax.add_patch(rect)
                
                if i < len(results['masks']):
                    mask = results['masks'][i, 0] > 0.5
                    masked = np.ma.masked_where(~mask, mask)
                    ax.imshow(masked, alpha=0.3, cmap='coolwarm')
                
                label_text = f"{fruit}-{ripeness}: {score:.2f}"
                ax.text(x1, y1-5, label_text,
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor=color_normalized, alpha=0.7),
                       fontsize=8, color='white', weight='bold')
            
            ax.set_xlim(0, 1200)
            ax.set_ylim(1200, 0)
            ax.axis('off')
            ax.set_title(f'Model {idx+1}: {len(results["boxes"])} detections\n({os.path.basename(model_path)})', 
                        fontsize=12, pad=15)
        
        plt.tight_layout()
        
        save_path = f"four_model_comparison_{os.path.basename(image_path).split('.')[0]}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f" Four-model comparison saved to {save_path}")
        
        plt.show()
        
        # Print detailed comparison
        print(f"\nDETAILED FOUR-MODEL COMPARISON:")
        print(f"{'='*60}")
        
        for idx, (results, model_path) in enumerate(zip(all_results, model_paths)):
            print(f"Model {idx+1} ({os.path.basename(model_path)}):")
            print(f"  Detections: {len(results['boxes'])}")
            if len(results['scores']) > 0:
                print(f"  Avg confidence: {np.mean(results['scores']):.3f}")
                print(f"  Score range: {results['scores'].min():.3f} - {results['scores'].max():.3f}")
            print()

        # Fruit type comparison across all models
        print(f"Fruit type comparison across all models:")
        all_fruits = set()
        summaries = []
        for detector, results in zip(detectors, all_results):
            summary = detector.get_summary(results)
            summaries.append(summary)
            all_fruits.update(summary['fruits_by_type'].keys())
        
        for fruit in sorted(all_fruits):
            counts = []
            for summary in summaries:
                count = summary['fruits_by_type'].get(fruit, {'total': 0})['total']
                counts.append(count)
            print(f"  {fruit.capitalize()}: " + " | ".join([f"M{i+1}={c}" for i, c in enumerate(counts)]))
        
        return all_results
    
    def compare_all_pairs(self, model_paths, image_path, score_threshold=0.1):
        """
        Compare all possible pairs of models (6 comparisons for 4 models)
        
        Args:
            model_paths: List of model paths
            image_path: Path to test image
            score_threshold: Minimum confidence score
        """
        import itertools
        
        print(f"\n{'='*80}")
        print(f"ALL PAIRWISE MODEL COMPARISONS: {os.path.basename(image_path)}")
        print(f"{'='*80}")
        
        pairs = list(itertools.combinations(enumerate(model_paths), 2))
        
        for (idx1, model1), (idx2, model2) in pairs:
            print(f"\n--- Comparing Model {idx1+1} vs Model {idx2+1} ---")
            self.compare_models_side_by_side(model1, model2, image_path, score_threshold)
    
    def get_summary(self, results):
        """
        Get a text summary of detection results
        
        Args:
            results: Detection results from detect()
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'total_detections': len(results['boxes']),
            'fruits_by_type': {},
            'fruits_by_ripeness': {'ripe': 0, 'unripe': 0, 'spoiled': 0},
            'average_confidence': np.mean(results['scores']) if len(results['scores']) > 0 else 0
        }
        
        for fruit, ripeness in zip(results['fruit_names'], results['ripeness_states']):
            if fruit not in summary['fruits_by_type']:
                summary['fruits_by_type'][fruit] = {
                    'total': 0,
                    'ripe': 0,
                    'unripe': 0,
                    'spoiled': 0
                }
            summary['fruits_by_type'][fruit]['total'] += 1
            summary['fruits_by_type'][fruit][ripeness] += 1
            
            if ripeness in summary['fruits_by_ripeness']:
                summary['fruits_by_ripeness'][ripeness] += 1
        
        return summary
    
    def print_summary(self, results):
        """Print a nice summary of detection results"""
        summary = self.get_summary(results)
        
        print("\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        print(f"Total fruits detected: {summary['total_detections']}")
        print(f"Average confidence: {summary['average_confidence']:.2%}")
        
        if summary['total_detections'] > 0:
            print("\nBy fruit type:")
            for fruit, counts in summary['fruits_by_type'].items():
                print(f"  {fruit.capitalize()}:")
                print(f"    Total: {counts['total']}")
                print(f"    Ripe: {counts['ripe']}")
                print(f"    Unripe: {counts['unripe']}")
                print(f"    Spoiled: {counts['spoiled']}")
            
            print("\nOverall ripeness distribution:")
            print(f"  Ripe: {summary['fruits_by_ripeness']['ripe']}")
            print(f"  Unripe: {summary['fruits_by_ripeness']['unripe']}")
            print(f"  Spoiled: {summary['fruits_by_ripeness']['spoiled']}")
        else:
            print("No fruits detected!")
        
        print("="*50 + "\n")
    
    def compare_thresholds(self, image_path, thresholds=[0.05, 0.1, 0.2, 0.3, 0.5]):
        """Compare detection results at different thresholds"""
        print(f"\nComparing thresholds for {image_path}:")
        print("-" * 50)
        
        for threshold in thresholds:
            results = self.detect(image_path, score_threshold=threshold)
            print(f"Threshold {threshold:4.2f}: {len(results['boxes']):3d} detections")
            if len(results['boxes']) > 0:
                print(f"              Avg confidence: {np.mean(results['scores']):.3f}")
    
    def process_directory(self, input_dir, output_dir=None, score_threshold=0.1):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing 1200x1200 images
            output_dir: Optional directory to save visualizations
            score_threshold: Minimum confidence score
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            import glob
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
            image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
        
        print(f"Found {len(image_files)} images to process")
        
        all_summaries = []
        
        for idx, image_path in enumerate(image_files):
            print(f"\nProcessing {idx+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            try:
                results = self.detect(image_path, score_threshold)
                self.print_summary(results)
                if output_dir:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    save_path = os.path.join(output_dir, f"{base_name}_detected.png")
                    self.visualize(results, save_path, show_masks=True)
                
                all_summaries.append({
                    'filename': os.path.basename(image_path),
                    'summary': self.get_summary(results)
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        return all_summaries


def demo_inference():
    """Demo function showing how to use the detector with 4 models"""
    
    print("="*60)
    print("FRUIT DETECTION INFERENCE DEMO - 4 MODEL SUPPORT")
    print("="*60)

    models_available = []
    for i in range(1, 5):  # Check for model1.pth through model4.pth
        model_path = f"model{i}.pth"
        if os.path.exists(model_path):
            models_available.append(model_path)
    
    if not models_available:
        print(" No model files found!")
        return
    
    print(f" Found models: {', '.join(models_available)}")
    
    test_images = ["testpic1.jpg", "testpic2.jpg", "testpic3.jpg", "testpic4.jpg"]
    available_images = [img for img in test_images if os.path.exists(img)]
    
    if not available_images:
        print(" No test images found!")
        return
    
    print(f" Found test images: {', '.join(available_images)}")
    
    # Four-model comparison if we have all 4 models
    if len(models_available) == 4:
        print(f"\n{'='*80}")
        print("FOUR-MODEL SIDE-BY-SIDE COMPARISON")
        print(f"{'='*80}")
        
        detector_helper = FruitDetector(models_available[0])
        
        for image_path in available_images:
            detector_helper.compare_four_models(models_available, image_path, score_threshold=0.1)
    
    # Pairwise comparisons if we have multiple models
    if len(models_available) >= 2:
        print(f"\n{'='*80}")
        print("PAIRWISE MODEL COMPARISONS")
        print(f"{'='*80}")
        
        detector_helper = FruitDetector(models_available[0])
        
        # Just do first image for pairwise to avoid too much output
        if available_images:
            detector_helper.compare_all_pairs(models_available, available_images[0], score_threshold=0.1)

    # Individual model testing
    for i, model_path in enumerate(models_available, 1):
        model_name = f"Model {i}"
        print(f"\n{'='*20} {model_name} INDIVIDUAL TESTING {'='*20}")

        detector = FruitDetector(model_path)
        
        for test_img in available_images:
            print(f"\n--- {model_name} on {test_img} ---")

            detector.compare_thresholds(test_img)
            
            results = detector.detect(test_img, score_threshold=0.1)
            detector.print_summary(results)
            
            save_path = f"{model_path.split('.')[0]}_{test_img.split('.')[0]}_individual.png"
            detector.visualize(results, save_path=save_path, 
                             title=f"{model_name} - {test_img}")

        # Batch processing
        input_directory = "ResizedExtendingPictures/Cherry/"
        if os.path.exists(input_directory):
            print(f"\n--- {model_name} Batch Processing ---")
            output_directory = f"detection_results_{model_path.split('.')[0]}/"
            
            summaries = detector.process_directory(
                input_directory, 
                output_directory,
                score_threshold=0.1
            )
            
            if summaries:
                print(f"\n{model_name} BATCH STATISTICS:")
                print("="*40)
                
                total_fruits = sum(s['summary']['total_detections'] for s in summaries)
                print(f"Total images processed: {len(summaries)}")
                print(f"Total fruits detected: {total_fruits}")
                if len(summaries) > 0:
                    print(f"Average fruits per image: {total_fruits/len(summaries):.1f}")

                fruit_totals = {}
                for s in summaries:
                    for fruit, counts in s['summary']['fruits_by_type'].items():
                        if fruit not in fruit_totals:
                            fruit_totals[fruit] = 0
                        fruit_totals[fruit] += counts['total']
                
                if fruit_totals:
                    print("\nFruit type distribution:")
                    for fruit, count in fruit_totals.items():
                        print(f"  {fruit.capitalize()}: {count}")


def quick_four_model_comparison(model_paths=None, image_path="testpic1.jpg"):
    """Quick function to compare four models on one image"""
    if model_paths is None:
        model_paths = ["model1.pth", "model2.pth", "model3.pth", "model4.pth"]
    
    # Check if all files exist
    missing_files = [p for p in model_paths + [image_path] if not os.path.exists(p)]
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return
    
    if len(model_paths) != 4:
        print(f"❌ Need exactly 4 models, got {len(model_paths)}")
        return
    
    detector = FruitDetector(model_paths[0])  # Just for the method
    detector.compare_four_models(model_paths, image_path)


def quick_comparison(model1_path="model1.pth", model2_path="model2.pth", image_path="testpic1.jpg"):
    """Quick function to compare two models on one image"""
    if not all(os.path.exists(p) for p in [model1_path, model2_path, image_path]):
        print("❌ Missing files for comparison")
        return
    
    detector = FruitDetector(model1_path)  # Just for the method
    detector.compare_models_side_by_side(model1_path, model2_path, image_path)


def analyze_model_performance(model_paths, test_images, score_threshold=0.1):
    """
    Comprehensive analysis of multiple models across multiple images
    
    Args:
        model_paths: List of model paths to compare
        test_images: List of test image paths
        score_threshold: Minimum confidence score
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL PERFORMANCE ANALYSIS")
    print(f"{'='*80}")
    
    # Check file existence
    missing_models = [m for m in model_paths if not os.path.exists(m)]
    missing_images = [i for i in test_images if not os.path.exists(i)]
    
    if missing_models:
        print(f"❌ Missing model files: {', '.join(missing_models)}")
        return
    if missing_images:
        print(f"❌ Missing image files: {', '.join(missing_images)}")
        return
    
    # Load all models
    detectors = []
    for i, model_path in enumerate(model_paths):
        print(f"Loading model {i+1}/{len(model_paths)}: {os.path.basename(model_path)}")
        detector = FruitDetector(model_path)
        detectors.append(detector)
    
    # Performance tracking
    model_stats = {i: {'total_detections': 0, 'total_confidence': 0, 'image_count': 0} 
                   for i in range(len(model_paths))}
    
    # Process each image with each model
    for img_idx, image_path in enumerate(test_images):
        print(f"\n{'='*60}")
        print(f"Processing image {img_idx+1}/{len(test_images)}: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        image_results = []
        
        for model_idx, detector in enumerate(detectors):
            print(f"\nModel {model_idx+1} ({os.path.basename(model_paths[model_idx])}):")
            results = detector.detect(image_path, score_threshold)
            image_results.append(results)
            
            # Update stats
            detections = len(results['boxes'])
            avg_conf = np.mean(results['scores']) if len(results['scores']) > 0 else 0
            
            model_stats[model_idx]['total_detections'] += detections
            model_stats[model_idx]['total_confidence'] += avg_conf
            model_stats[model_idx]['image_count'] += 1
            
            print(f"  Detections: {detections}")
            print(f"  Avg confidence: {avg_conf:.3f}")
        
        # Create comparison visualization for this image
        if len(model_paths) == 4:
            detector.compare_four_models(model_paths, image_path, score_threshold)
        elif len(model_paths) == 2:
            detector.compare_models_side_by_side(model_paths[0], model_paths[1], image_path, score_threshold)
    
    # Final performance summary
    print(f"\n{'='*80}")
    print("FINAL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    for i, model_path in enumerate(model_paths):
        stats = model_stats[i]
        avg_detections = stats['total_detections'] / stats['image_count'] if stats['image_count'] > 0 else 0
        avg_confidence = stats['total_confidence'] / stats['image_count'] if stats['image_count'] > 0 else 0
        
        print(f"\nModel {i+1} ({os.path.basename(model_path)}):")
        print(f"  Total detections across all images: {stats['total_detections']}")
        print(f"  Average detections per image: {avg_detections:.1f}")
        print(f"  Average confidence per image: {avg_confidence:.3f}")
        print(f"  Images processed: {stats['image_count']}")
    
    # Ranking
    print(f"\n{'='*40}")
    print("MODEL RANKINGS")
    print(f"{'='*40}")
    
    # Rank by average detections per image
    detection_ranking = sorted(model_stats.items(), 
                              key=lambda x: x[1]['total_detections'] / max(x[1]['image_count'], 1), 
                              reverse=True)
    
    print("By average detections per image:")
    for rank, (model_idx, stats) in enumerate(detection_ranking, 1):
        avg_det = stats['total_detections'] / max(stats['image_count'], 1)
        print(f"  {rank}. Model {model_idx+1}: {avg_det:.1f} detections/image")
    
    # Rank by average confidence
    confidence_ranking = sorted(model_stats.items(), 
                               key=lambda x: x[1]['total_confidence'] / max(x[1]['image_count'], 1), 
                               reverse=True)
    
    print("\nBy average confidence:")
    for rank, (model_idx, stats) in enumerate(confidence_ranking, 1):
        avg_conf = stats['total_confidence'] / max(stats['image_count'], 1)
        print(f"  {rank}. Model {model_idx+1}: {avg_conf:.3f} avg confidence")


if __name__ == "__main__":
    # demo_inference()
    
    # Uncomment these for quick comparisons:
    # quick_comparison("model1.pth", "model2.pth", "testpic1.jpg")
    # quick_four_model_comparison(["model1.pth", "model2.pth", "model3.pth", "model4.pth"], "testpic1.jpg")
    
    # Uncomment for comprehensive analysis:
    analyze_model_performance(
        ["model1.pth", "model2.pth", "model3.pth", "model4.pth"],
        ["testpic1.jpg", "testpic2.jpg", "testpic3.jpg", "testpic4.jpg", "testpic5.jpg", "testpic6.jpg", "testpic7.jpg", "testpic8.jpg", "testpic9.jpg", "testpic10.jpg"]
    )