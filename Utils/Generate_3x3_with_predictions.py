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
    
    def compare_models_by_rows(self, model_paths, image_paths, score_threshold=0.1, images_per_row=3):
        """
        Compare multiple models showing each model's results in a row
        
        Args:
            model_paths: List of model paths (e.g., ["model2.pth", "model3.pth", "model4.pth"])
            image_paths: List of test image paths (e.g., ["testpic1.jpg", "testpic2.jpg", "testpic3.jpg"])
            score_threshold: Minimum confidence score
            images_per_row: Number of images to show per row (default 3)
        """
        num_models = len(model_paths)
        num_images = min(len(image_paths), images_per_row)
        
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON BY ROWS")
        print(f"Models: {num_models}, Images per model: {num_images}")
        print(f"{'='*80}")
        
        # Create figure with subplots
        fig, axes = plt.subplots(num_models, num_images, figsize=(8*num_images, 8*num_models))
        
        # Handle single row/column cases
        if num_models == 1:
            axes = axes.reshape(1, -1)
        if num_images == 1:
            axes = axes.reshape(-1, 1)
        
        # Process each model
        for model_idx, model_path in enumerate(model_paths):
            print(f"\nProcessing Model {model_idx+1}: {os.path.basename(model_path)}")
            
            # Load the model
            detector = FruitDetector(model_path)
            
            # Process each image for this model
            for img_idx in range(num_images):
                if img_idx < len(image_paths):
                    image_path = image_paths[img_idx]
                    ax = axes[model_idx, img_idx]
                    
                    # Detect fruits
                    results = detector.detect(image_path, score_threshold)
                    
                    # Draw the image and detections
                    image = results['image'].copy()
                    ax.imshow(image)
                    
                    # Draw boxes and labels
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
                        
                        # Add mask if available
                        if i < len(results['masks']):
                            mask = results['masks'][i, 0] > 0.5
                            masked = np.ma.masked_where(~mask, mask)
                            ax.imshow(masked, alpha=0.3, cmap='coolwarm')
                        
                        # Add label
                        label_text = f"{fruit}-{ripeness}: {score:.2f}"
                        ax.text(x1, y1-5, label_text,
                               bbox=dict(boxstyle="round,pad=0.3", 
                                        facecolor=color_normalized, alpha=0.7),
                               fontsize=8, color='white', weight='bold')
                    
                    ax.set_xlim(0, 1200)
                    ax.set_ylim(1200, 0)
                    ax.axis('off')
                    
                    # Set title
                    if model_idx == 0:
                        ax.set_title(f'{os.path.basename(image_path)}\n{len(results["boxes"])} detections', 
                                   fontsize=12, pad=10)
                    else:
                        ax.set_title(f'{len(results["boxes"])} detections', fontsize=12, pad=10)
                    
                    # Add model label on the left
                    if img_idx == 0:
                        ax.text(-0.15, 0.5, f'Model {model_idx+1}\n({os.path.basename(model_path)})', 
                               transform=ax.transAxes,
                               fontsize=14, weight='bold',
                               ha='right', va='center',
                               rotation=90)
        
        plt.tight_layout()
        
        # Save the figure
        save_path = f"model_comparison_rows_{num_models}models_{num_images}images.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nComparison saved to {save_path}")
        
        plt.show()
        
        return fig
    
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


def demo_row_comparison():
    """Demo function showing the new row-based comparison"""
    
    print("="*60)
    print("FRUIT DETECTION - ROW-BASED MODEL COMPARISON")
    print("="*60)
    
    # Define models to compare (models 2, 3, and 4)
    model_paths = ["model2.pth", "model3.pth", "model4.pth"]
    
    # Check which models are available
    available_models = [m for m in model_paths if os.path.exists(m)]
    
    if not available_models:
        print(" No model files found!")
        return
    
    print(f" Found models: {', '.join(available_models)}")
    
    # Define test images
    test_images = ["testpic1.jpg", "testpic2.jpg", "testpic3.jpg"]
    available_images = [img for img in test_images if os.path.exists(img)]
    
    if not available_images:
        print(" No test images found!")
        return
    
    print(f" Found test images: {', '.join(available_images)}")
    
    # Use the first available model to access the comparison method
    detector = FruitDetector(available_models[0])
    
    # Create the row-based comparison
    detector.compare_models_by_rows(available_models, available_images, score_threshold=0.1)
    
    # You can also create different configurations
    # For example, showing all 10 test pictures for each model (in multiple figures)
    all_test_images = [f"testpic{i}.jpg" for i in range(1, 11)]
    available_all_images = [img for img in all_test_images if os.path.exists(img)]
    
    if len(available_all_images) > 3:
        print("\n" + "="*60)
        print("Creating additional comparisons with more images...")
        print("="*60)
        
        # Show first 5 images
        if len(available_all_images) >= 5:
            detector.compare_models_by_rows(available_models, available_all_images[:5], 
                                          score_threshold=0.1, images_per_row=5)
        
        # Show next 5 images
        if len(available_all_images) >= 10:
            detector.compare_models_by_rows(available_models, available_all_images[5:10], 
                                          score_threshold=0.1, images_per_row=5)


if __name__ == "__main__":
    # demo_row_comparison()
    
    # You can also call it directly with custom parameters:
    detector = FruitDetector("model2.pth")
    detector.compare_models_by_rows(
        ["model2.pth", "model3.pth", "model4.pth"],
        ["testpic1.jpg", "testpic2.jpg", "testpic3.jpg"],
        score_threshold=0.1,
        images_per_row=3
    )