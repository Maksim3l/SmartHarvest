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
        
        # Class names mapping
        self.class_names = ['background',
                           'apple-ripe', 'apple-unripe', 'apple-spoiled',
                           'cherry-ripe', 'cherry-unripe', 'cherry-spoiled',
                           'cucumber-ripe', 'cucumber-unripe', 'cucumber-spoiled',
                           'strawberry-ripe', 'strawberry-unripe', 'strawberry-spoiled',
                           'tomato-ripe', 'tomato-unripe', 'tomato-spoiled']
        
        # Colors for visualization (per fruit type)
        self.fruit_colors = {
            'apple': (255, 0, 0),      # Red
            'cherry': (139, 0, 0),     # Dark red
            'cucumber': (0, 255, 0),   # Green
            'strawberry': (255, 20, 147),  # Pink
            'tomato': (255, 69, 0)     # Orange-red
        }
        
        # Load model
        self._load_model()
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """Load the trained model"""
        print(f"Loading model from {self.model_path}...")
        
        # Create model architecture
        self.model = maskrcnn_resnet50_fpn(weights=None)
        
        # Replace heads
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 16)
        
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 16)
        
        # Load weights
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print("✓ Model loaded successfully!")
    
    def detect(self, image_path, score_threshold=0.5):
        """
        Detect fruits in an image
        
        Args:
            image_path: Path to input image (should be 1200x1200)
            score_threshold: Minimum confidence score for detections
            
        Returns:
            dict: Detection results with boxes, labels, scores, masks
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        # Check size
        h, w = image.shape[:2]
        if h != 1200 or w != 1200:
            print(f"Warning: Image size is {w}x{h}, expected 1200x1200")
            print("Resizing to 1200x1200...")
            image = cv2.resize(image, (1200, 1200))
        
        # Prepare image
        image_tensor = self.transform(Image.fromarray(image))
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Run detection
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]
        
        # Filter by score
        keep = predictions['scores'] > score_threshold
        
        # Extract results
        results = {
            'boxes': predictions['boxes'][keep].cpu().numpy(),
            'labels': predictions['labels'][keep].cpu().numpy(),
            'scores': predictions['scores'][keep].cpu().numpy(),
            'masks': predictions['masks'][keep].cpu().numpy(),
            'image': image
        }
        
        # Add fruit names and ripeness
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
    
    def visualize(self, results, save_path=None, show_masks=True):
        """
        Visualize detection results
        
        Args:
            results: Detection results from detect()
            save_path: Optional path to save visualization
            show_masks: Whether to show segmentation masks
        """
        image = results['image'].copy()
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        
        # Draw detections
        for i in range(len(results['boxes'])):
            box = results['boxes'][i]
            label = results['labels'][i]
            score = results['scores'][i]
            fruit = results['fruit_names'][i]
            ripeness = results['ripeness_states'][i]
            
            # Get color for fruit type
            color = self.fruit_colors.get(fruit, (255, 255, 255))
            color_normalized = [c/255.0 for c in color]
            
            # Draw bounding box
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, edgecolor=color_normalized, 
                               linewidth=3)
            ax.add_patch(rect)
            
            # Draw mask if requested
            if show_masks and i < len(results['masks']):
                mask = results['masks'][i, 0] > 0.5
                masked = np.ma.masked_where(~mask, mask)
                ax.imshow(masked, alpha=0.3, cmap='coolwarm')
            
            # Add label
            label_text = f"{fruit}-{ripeness}: {score:.2f}"
            ax.text(x1, y1-5, label_text,
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=color_normalized, alpha=0.7),
                   fontsize=12, color='white', weight='bold')
        
        ax.set_xlim(0, 1200)
        ax.set_ylim(1200, 0)
        ax.axis('off')
        ax.set_title(f'Detected {len(results["boxes"])} fruits', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved to {save_path}")
        
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
        
        # Count by fruit type and ripeness
        for fruit, ripeness in zip(results['fruit_names'], results['ripeness_states']):
            # By type
            if fruit not in summary['fruits_by_type']:
                summary['fruits_by_type'][fruit] = {
                    'total': 0,
                    'ripe': 0,
                    'unripe': 0,
                    'spoiled': 0
                }
            summary['fruits_by_type'][fruit]['total'] += 1
            summary['fruits_by_type'][fruit][ripeness] += 1
            
            # By ripeness
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
        print("="*50 + "\n")
    
    def process_directory(self, input_dir, output_dir=None, score_threshold=0.5):
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing 1200x1200 images
            output_dir: Optional directory to save visualizations
            score_threshold: Minimum confidence score
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Find all images
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
                # Detect
                results = self.detect(image_path, score_threshold)
                
                # Print summary
                self.print_summary(results)
                
                # Save visualization if requested
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
    """Demo function showing how to use the detector"""
    
    # Path to your trained model
    model_path = "fruit_detection_model/best_model.pth"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train the model first using the training script.")
        return
    
    # Initialize detector
    detector = FruitDetector(model_path)
    
    # Example 1: Detect in a single image
    print("\n" + "="*50)
    print("EXAMPLE 1: Single Image Detection")
    print("="*50)
    
    # You would replace this with your actual image path
    test_image_path = "RePictures/apple/apple_1.jpg"  # Example
    
    if os.path.exists(test_image_path):
        results = detector.detect(test_image_path, score_threshold=0.5)
        detector.print_summary(results)
        detector.visualize(results, save_path="detection_result.png")
    else:
        print("Test image not found. Creating a dummy example...")
        # Create a black 1200x1200 image as example
        dummy_image = np.zeros((1200, 1200, 3), dtype=np.uint8)
        results = detector.detect(dummy_image, score_threshold=0.5)
        detector.print_summary(results)
    
    # Example 2: Process entire directory
    print("\n" + "="*50)
    print("EXAMPLE 2: Batch Processing")
    print("="*50)
    
    input_directory = "RePictures/"  # Your test images directory
    output_directory = "detection_results/"
    
    if os.path.exists(input_directory):
        summaries = detector.process_directory(
            input_directory, 
            output_directory,
            score_threshold=0.5
        )
        
        # Print overall statistics
        print("\n" + "="*50)
        print("OVERALL STATISTICS")
        print("="*50)
        
        total_fruits = sum(s['summary']['total_detections'] for s in summaries)
        print(f"Total images processed: {len(summaries)}")
        print(f"Total fruits detected: {total_fruits}")
        print(f"Average fruits per image: {total_fruits/len(summaries):.1f}")


if __name__ == "__main__":
    demo_inference()