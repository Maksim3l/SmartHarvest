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

def debug_model_predictions():
    """Debug why the model isn't making predictions"""
    
    print("="*60)
    print("DEBUGGING MODEL PREDICTIONS")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = maskrcnn_resnet50_fpn(weights=None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 16)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 16)
    
    checkpoint = torch.load('model2.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")
    print(f"✓ Best score: {checkpoint['best_score']}")
    
    print("\n" + "-"*40)
    print("TEST 1: Random tensor input")
    print("-"*40)
    
    dummy_input = torch.randn(1, 3, 1200, 1200).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)
    
    print(f"Random input predictions:")
    print(f"  Boxes: {dummy_output[0]['boxes'].shape}")
    print(f"  Scores: {dummy_output[0]['scores'].shape}")
    
    if len(dummy_output[0]['scores']) > 0:
        print(f"  Score range: {dummy_output[0]['scores'].min():.6f} to {dummy_output[0]['scores'].max():.6f}")
        print(f"  Scores > 0.01: {(dummy_output[0]['scores'] > 0.01).sum()}")
        print(f"  Scores > 0.001: {(dummy_output[0]['scores'] > 0.001).sum()}")
    else:
        print("   NO PREDICTIONS AT ALL! This indicates a broken model.")

    print("\n" + "-"*40)
    print("TEST 2: Real image - NO normalization")
    print("-"*40)
    
    test_images = ["testpic1.jpg", "testpic2.jpg"]
    
    for test_img in test_images:
        if not os.path.exists(test_img):
            print(f" {test_img} not found")
            continue
            
        print(f"\nTesting {test_img}:")

        image = cv2.imread(test_img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        print(f"  Original size: {w}x{h}")
        
        if h != 1200 or w != 1200:
            image = cv2.resize(image, (1200, 1200))
            print("  Resized to 1200x1200")
        
        transform_no_norm = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform_no_norm(Image.fromarray(image))
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        print(f"  Tensor shape: {image_tensor.shape}")
        print(f"  Tensor range: {image_tensor.min():.3f} to {image_tensor.max():.3f}")
        
        with torch.no_grad():
            predictions = model(image_tensor)[0]
        
        print(f"  Total predictions: {len(predictions['boxes'])}")
        
        if len(predictions['scores']) > 0:
            print(f"  Score range: {predictions['scores'].min():.6f} to {predictions['scores'].max():.6f}")
            print(f"  Top 10 scores: {predictions['scores'][:10].cpu().numpy()}")
            print(f"  Scores > 0.1: {(predictions['scores'] > 0.1).sum()}")
            print(f"  Scores > 0.01: {(predictions['scores'] > 0.01).sum()}")
            print(f"  Scores > 0.001: {(predictions['scores'] > 0.001).sum()}")
            
            top_labels = predictions['labels'][:10].cpu().numpy()
            print(f"  Top 10 labels: {top_labels}")
        else:
            print("   NO PREDICTIONS for real image!")
    
    print("\n" + "-"*40)
    print("TEST 3: Real image - WITH ImageNet normalization")
    print("-"*40)
    
    transform_with_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if os.path.exists("testpic1.jpg"):
        image = cv2.imread("testpic1.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (1200, 1200):
            image = cv2.resize(image, (1200, 1200))
        
        image_tensor_norm = transform_with_norm(Image.fromarray(image))
        image_tensor_norm = image_tensor_norm.unsqueeze(0).to(device)
        
        print(f"  Normalized tensor range: {image_tensor_norm.min():.3f} to {image_tensor_norm.max():.3f}")
        
        with torch.no_grad():
            predictions_norm = model(image_tensor_norm)[0]
        
        print(f"  WITH normalization:")
        if len(predictions_norm['scores']) > 0:
            print(f"    Score range: {predictions_norm['scores'].min():.6f} to {predictions_norm['scores'].max():.6f}")
            print(f"    Scores > 0.01: {(predictions_norm['scores'] > 0.01).sum()}")
            print(f"    Top 5 scores: {predictions_norm['scores'][:5].cpu().numpy()}")
        else:
            print("     NO PREDICTIONS with normalization either!")
    
    print("\n" + "-"*40)
    print("TEST 4: Model weight sanity check")
    print("-"*40)
    
    conv1_weights = model.backbone.body.conv1.weight
    print(f"  First conv layer weights:")
    print(f"    Shape: {conv1_weights.shape}")
    print(f"    Range: {conv1_weights.min():.6f} to {conv1_weights.max():.6f}")
    print(f"    Mean: {conv1_weights.mean():.6f}")
    print(f"    Std: {conv1_weights.std():.6f}")
    
    cls_weights = model.roi_heads.box_predictor.cls_score.weight
    print(f"  Classifier weights:")
    print(f"    Shape: {cls_weights.shape}")
    print(f"    Range: {cls_weights.min():.6f} to {cls_weights.max():.6f}")
    
    print("\n" + "-"*40)
    print("TEST 5: Check RPN separately")
    print("-"*40)
    
    if os.path.exists("testpic1.jpg"):
        image = cv2.imread("testpic1.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[:2] != (1200, 1200):
            image = cv2.resize(image, (1200, 1200))
        
        transform_no_norm = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform_no_norm(Image.fromarray(image))
        image_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            features = model.backbone(image_tensor)
            rpn_features = [features[f] for f in features.keys()]
            
            model.rpn.training = False
            proposals, proposal_losses = model.rpn(image_tensor, features, None)
            
            print(f"  RPN generated {len(proposals[0])} proposals")
            if len(proposals[0]) > 0:
                print(f"  Proposal scores range: {proposals[0].objectness_logits.min():.6f} to {proposals[0].objectness_logits.max():.6f}")
            else:
                print("   RPN generated NO proposals - this is the root cause!")
    
    print("\n" + "="*60)

def test_different_thresholds():
    """Test the model with very low thresholds"""
    print("Testing with different score thresholds...")
    
    from UseModelInference import FruitDetector

    detector = FruitDetector("model2.pth")
    detector.transform = transforms.Compose([transforms.ToTensor()])
    
    if os.path.exists("testpic1.jpg"):
        thresholds = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
        
        for threshold in thresholds:
            results = detector.detect("testpic1.jpg", score_threshold=threshold)
            print(f"  Threshold {threshold:5.3f}: {len(results['boxes'])} detections")
            if len(results['boxes']) > 0:
                print(f"    Scores: {results['scores'][:3]}")

def test_model1_vs_model2():
    """Compare both model files"""
    print("\n" + "="*60)
    print("COMPARING MODEL1.PTH vs MODEL2.PTH")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if os.path.exists("model1.pth"):
        print("\n--- Testing model1.pth ---")
        try:
            model1 = maskrcnn_resnet50_fpn(weights=None)
            in_features = model1.roi_heads.box_predictor.cls_score.in_features
            model1.roi_heads.box_predictor = FastRCNNPredictor(in_features, 16)
            in_features_mask = model1.roi_heads.mask_predictor.conv5_mask.in_channels
            model1.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 16)
            
            state_dict1 = torch.load('model1.pth', map_location=device)
            model1.load_state_dict(state_dict1)
            model1.to(device)
            model1.eval()
            
            dummy_input = torch.randn(1, 3, 1200, 1200).to(device)
            with torch.no_grad():
                output1 = model1(dummy_input)[0]
            
            print(f"  Model 1 predictions: {len(output1['boxes'])}")
            if len(output1['scores']) > 0:
                print(f"  Score range: {output1['scores'].min():.6f} to {output1['scores'].max():.6f}")
            else:
                print("   Model 1 also produces no predictions!")
                
        except Exception as e:
            print(f"   Error loading model1.pth: {e}")
    else:
        print("model1.pth not found")

    print("\n--- Re-testing model2.pth ---")
    try:
        model2 = maskrcnn_resnet50_fpn(weights=None)
        in_features = model2.roi_heads.box_predictor.cls_score.in_features
        model2.roi_heads.box_predictor = FastRCNNPredictor(in_features, 16)
        in_features_mask = model2.roi_heads.mask_predictor.conv5_mask.in_channels
        model2.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 16)
        
        checkpoint2 = torch.load('model2.pth', map_location=device)
        model2.load_state_dict(checkpoint2['model_state_dict'])
        model2.to(device)
        model2.eval()
        
        dummy_input = torch.randn(1, 3, 1200, 1200).to(device)
        with torch.no_grad():
            output2 = model2(dummy_input)[0]
        
        print(f"  Model 2 predictions: {len(output2['boxes'])}")
        if len(output2['scores']) > 0:
            print(f"  Score range: {output2['scores'].min():.6f} to {output2['scores'].max():.6f}")
        else:
            print("   Model 2 produces no predictions!")
            
    except Exception as e:
        print(f"   Error with model2.pth: {e}")

if __name__ == "__main__":
    debug_model_predictions()
    test_model1_vs_model2()
    print("\n")
    # test_different_thresholds()  # Skip this since model produces no outputs