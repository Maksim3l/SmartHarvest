# Hybrid Semantic-Instance Segmentation for Fruit Quality Assessment in Garden Environments

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A unified deep learning framework for multi-class fruit quality assessment in natural agricultural environments using Mask R-CNN with ResNet-50 backbone and Feature Pyramid Networks.

## Overview

This repository contains the implementation of a hybrid semantic-instance segmentation approach capable of simultaneously detecting, segmenting, and classifying fruits from five species (apple, cherry, cucumber, strawberry, tomato), each with three distinct ripeness states (unripe, ripe, spoiled), yielding 15 total fruit-quality combinations.

**Key Features:**
- Instance segmentation with pixel-level precision
- Multi-class fruit species and ripeness classification
- Robust performance in complex agricultural environments
- Custom training pipeline with extensive data augmentation
- Support for overlapping fruit detection

## Architecture

The model is based on Mask R-CNN with:
- **Backbone**: ResNet-50 pre-trained on MS-COCO
- **Feature Extraction**: Feature Pyramid Network (FPN)
- **Classification Head**: FastRCNNPredictor (16 classes including background)
- **Mask Head**: MaskRCNNPredictor with 256 hidden units
- **Custom modifications**: Optimized anchor sizes and NMS thresholds for fruit detection

## Installation

### Requirements
```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- Python 3.8+
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- OpenCV >= 4.5.0
- numpy
- matplotlib
- Pillow
- albumentations

### Clone Repository
```bash
git clone https://github.com/Maksim3l/SmartHarvest
cd fruit-quality-assessment
```

## Dataset

The dataset consists of 487 high-resolution images with polygon annotations, captured in diverse natural garden conditions. Images are resized and padded to 1200×1200 pixels.

**Dataset Statistics:**
- 5 fruit species: apple, cherry, cucumber, strawberry, tomato
- 3 ripeness states per species: unripe, ripe, spoiled
- 15 total classes + background
- Manual polygon annotations using VGG Image Annotator (VIA)
- 80/20 train/validation split

Download the dataset from [Hugging Face](TheCoffeeAddict/SmartHarvest).

## Training

### Quick Start
```bash
python training.py
```

### Training Configuration
The training pipeline includes:
- **Optimizer**: SGD with momentum (0.9)
- **Learning Rate**: 2×10⁻³ with warm-up and exponential decay
- **Batch Size**: 2 (due to no GPU memory (CUDA problems))
- **Epochs**: Up to 100 with early stopping (patience=15)
- **Regularization**: Dropout (0.3), weight decay (5×10⁻⁴)
- **Mixed Precision**: Enabled for memory optimization

### Data Augmentation
Extensive augmentation pipeline includes:
- Photometric: brightness, contrast, saturation, hue adjustments
- Geometric: horizontal flipping, rotation (±15°), translation (±10%)
- 8x dataset expansion (3,896 synthetic training images)

### Custom Training
```bash
python Training.py or Training_without.py
```

## Inference

```bash
python Models/UseModelInference.py
```

## Model Performance

The final model demonstrates:
- Accurate instance boundary detection
- Effective separation of overlapping fruits
- Multi-class species and ripeness classification
- Reduced false positive rates compared to earlier iterations
- Stable convergence within 35-45 epochs

**Loss Function:**
```
L_total = L_cls + L_box + L_mask
```

## Results

The model shows strong performance across different fruit types and environmental conditions:
- Robust detection in varying lighting conditions
- Accurate segmentation of overlapping instances
- Reliable classification of ripeness states
- Some challenges in tomato-strawberry differentiation

## Applications

This system enables:
- **Robotic Harvesting**: Instance-level fruit localization for grasp planning
- **Quality Assessment**: Real-time ripeness evaluation
- **Yield Prediction**: Automated crop monitoring
- **Precision Agriculture**: Species-specific management strategies

## Future Work

- **Dataset Expansion**: Pseudo-labeling pipeline for 7,000+ unlabeled images
- **Mobile Deployment**: Model optimization for edge devices
- **Additional Species**: Extension to pears, plums, raspberries, peppers
- **3D Integration**: Depth sensor fusion for spatial localization
- **Video Processing**: Temporal consistency modeling

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{loknar2025hybrid,
    title={Hybrid Semantic-Instance Segmentation for Fruit Quality Assessment in Garden Environments},
    author={Loknar Maksim, Mlakar Uroš},
    booktitle={Student Computing Research Symposium},
    year={2025},
    organization={University of Maribor}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Jan Popič for help with establishing the experimental environment
- Faculty of Electrical Engineering and Computer Science, University of Maribor
- VGG Image Annotator team for the annotation tool

## Contact

- **Maksim Loknar**
- **Uroš Mlakar**

---

**University of Maribor, Faculty of Electrical Engineering and Computer Science**