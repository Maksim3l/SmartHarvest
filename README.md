# SmartHarvest: Comprehensive Multi-Species Fruit Ripeness Detection and Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-SCORES'25-red.svg)](https://arxiv.org/abs/XXXX.XXXX)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/TheCoffeeAddict/SmartHarvest)

**Authors:** [Maksim Loknar](mailto:maksim.loknar@student.um.si), [Uroš Mlakar](mailto:uros.mlakar@um.si)  
**Institution:** Faculty of Electrical Engineering and Computer Science, University of Maribor, Slovenia  
**Conference:** Student Computing Research Symposium (SCORES) 2025

---

## 🎯 Overview

SmartHarvest presents a comprehensive framework for constructing large-scale multi-species fruit ripeness detection datasets and validates it through focused instance segmentation. Our work addresses critical gaps in agricultural computer vision by establishing consistent annotation protocols across diverse fruit morphologies while maintaining species-specific ripeness criteria.

### 🔬 Research Contributions

- **Comprehensive Dataset**: 8 fruit species with 6,984 polygon-based annotations across 5 ripeness states
- **Scalable Methodology**: Flexible annotation protocols accommodating diverse fruit morphologies  
- **Rigorous Quality Control**: Cross-species consistency with expert agricultural validation
- **Practical Validation**: Instance segmentation on apple-cherry subsets achieving 22.49% AP@0.5

### 📊 Key Results

| Metric | Apple-Cherry Model | Multi-Species Dataset |
|--------|-------------------|---------------------|
| **AP@0.5** | **22.49%** | 8 species coverage |
| **COCO mAP** | **60.63%** | 6,984 annotations |
| **Training Images** | 2,240 (15x augmented) | 486 base images |
| **Species Coverage** | Apple, Cherry focus | 8+ species expanding |

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Maksim3l/SmartHarvest.git
cd SmartHarvest
pip install -r requirements.txt
```

### Dataset Download

```python
# From Hugging Face
from datasets import load_dataset
dataset = load_dataset("TheCoffeeAddict/SmartHarvest")

# Or download directly
wget https://huggingface.co/datasets/TheCoffeeAddict/SmartHarvest/resolve/main/dataset.zip
```

### Training

```bash
# Apple-Cherry focused detection
python Training.py --config configs/apple_cherry_focused.yaml

# Multi-species training (coming soon)
python Training.py --config configs/eight_species.yaml
```

### Inference

```python
from Models.UseModelInference import FruitDetector

detector = FruitDetector("path/to/trained/model.pth")
results = detector.predict("path/to/image.jpg")
```

---

## 📁 Project Structure

```
SmartHarvest/
├── README.md
├── requirements.txt
├── CITATION.cff
├── configs/
│   ├── apple_cherry_focused.yaml    # Focused training config
│   └── eight_species.yaml           # Multi-species config
├── data/
│   ├── raw/                         # Original images
│   ├── processed/                   # Preprocessed datasets
│   └── annotations/                 # VIA format annotations
├── Models/
│   ├── UseModelInference.py         # Inference pipeline
│   └── architecture.py             # Model definitions
├── Training.py                      # Main training script
├── Training_without.py              # Alternative training
├── notebooks/                       # Exploratory analysis
└── results/
    ├── figures/                     # Publication figures
    └── checkpoints/                 # Trained models
```

---

## 🍎 Dataset Overview

### Species and Ripeness Coverage

| Species | Images | Annotations | Ripeness States |
|---------|--------|-------------|-----------------|
| **Apple** | 98 | 2,582 | Ripe (79.7%), Unripe (10.8%), Spoiled (9.3%) |
| **Cherry** | 86 | 969 | Ripe (49.6%), Unripe (37.4%), Spoiled (13.0%) |
| **Tomato** | 94 | 1,572 | Ripe (43.7%), Unripe (51.9%), Spoiled (4.4%) |
| **Strawberry** | 111 | 1,397 | Ripe (42.3%), Unripe (53.7%), Spoiled (4.1%) |
| **Cucumber** | 97 | 464 | Ripe (77.1%), Unripe (19.8%), Spoiled (3.1%) |
| **Total** | **486** | **6,984** | **3 states + obscured category** |

### Annotation Statistics

- **Average annotations per image**: 14.4
- **Polygon vertices**: 14.1 ± 9.8 (range: 3-126)
- **Obscured instances**: 53.8% (realistic occlusion challenge)
- **Mean bounding box**: 71.4 × 78.8 pixels
- **Annotation tool**: VGG Image Annotator (VIA)

---

## 🏗️ Model Architecture

### Mask R-CNN Configuration

- **Backbone**: ResNet-50 + Feature Pyramid Networks (FPN)
- **Pre-training**: MS-COCO weights for initialization
- **Classes**: 7 (apple-ripe, apple-unripe, apple-spoiled, cherry-ripe, cherry-unripe, cherry-spoiled, background)
- **Anchor scales**: [8, 16, 32, 64, 128] pixels (optimized for fruit sizes)

### Training Configuration

```python
# Key hyperparameters
optimizer = SGD(momentum=0.9, weight_decay=5e-4)
learning_rate = 0.002  # with 5-epoch warmup
batch_size = 2
max_epochs = 100
early_stopping_patience = 15
```

### Performance Breakdown

| Class | AP@0.5 | Precision | Recall |
|-------|--------|-----------|--------|
| Apple-Ripe | 10.45% | - | - |
| Apple-Unripe | 25.00% | - | - |
| Apple-Spoiled | **32.60%** | - | - |
| Cherry-Ripe | 18.20% | - | - |
| Cherry-Unripe | 17.10% | - | - |
| Cherry-Spoiled | **31.56%** | - | - |

*Note: Spoiled fruits achieve highest detection accuracy due to distinctive visual features*

---

## 🔬 Experimental Results

### Detection Performance

- **AP@0.5**: 22.49% (standard IoU threshold)
- **AP@0.75**: 7.98% (strict localization)
- **COCO mAP**: 60.63% (across IoU thresholds 0.5-0.95)
- **Total detections**: 1,393 (325 true positives at IoU≥0.5)

### Key Insights

1. **Spoiled fruit detection** outperforms ripe/unripe classification
2. **Balanced species performance** (Apple: 22.68%, Cherry: 22.29%)
3. **Occlusion challenge** significant with 53.8% partially obscured fruits
4. **Real-world applicability** demonstrated in garden environments

---

## 📚 Usage Examples

### Training Custom Model

```python
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load configuration
config = yaml.load(open("configs/apple_cherry_focused.yaml"))

# Initialize model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.roi_heads.box_predictor = FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, 
    num_classes=7
)

# Train
trainer = FruitTrainer(model, config)
trainer.train(train_loader, val_loader)
```

### Inference Pipeline

```python
from Models.UseModelInference import FruitDetector
import cv2

# Initialize detector
detector = FruitDetector("checkpoints/best_model.pth")

# Process image
image = cv2.imread("test_image.jpg")
results = detector.predict(image)

# Extract results
for detection in results:
    bbox = detection['bbox']
    mask = detection['mask'] 
    species = detection['species']
    ripeness = detection['ripeness']
    confidence = detection['score']
```

---

## 🎯 Applications

### Precision Agriculture
- **Automated harvesting** with ripeness-aware robot guidance
- **Yield prediction** through early fruit counting and quality assessment
- **Quality control** in commercial orchards and packing facilities

### Research Extensions
- **Temporal modeling** for ripeness progression prediction
- **Multi-modal fusion** with hyperspectral or depth sensors
- **Mobile deployment** for handheld assessment tools

---

## 📈 Future Development

### Dataset Expansion
- **Target**: 500+ images per species for production deployment
- **New species**: Plums (in progress), grapes, peaches, lettuce, carrots
- **Geographic diversity**: Multi-region data collection to reduce bias
- **Temporal coverage**: Full seasonal cycles for each species

### Model Improvements
- **Class-balanced training** to address ripeness state imbalances
- **Agricultural pre-training** beyond MS-COCO initialization  
- **Multi-scale architectures** for small fruit detection (berries)
- **Uncertainty estimation** for confidence-aware deployment

### Technical Roadmap
- **Semi-supervised learning** with 7,000+ unlabeled images
- **Active learning** for efficient annotation expansion
- **Model compression** for mobile/edge deployment
- **Real-time inference** optimization for robotic systems

---

## 📖 Citation

If you use SmartHarvest in your research, please cite our work:

```bibtex
@inproceedings{loknar2025comprehensive,
    title={Comprehensive Multi-Species Fruit Ripeness Dataset Construction: From Eight-Species Collection to Focused Apple-Cherry Detection},
    author={Loknar, Maksim and Mlakar, Uroš},
    booktitle={Student Computing Research Symposium},
    year={2025},
    organization={University of Maribor},
    url={https://github.com/Maksim3l/SmartHarvest}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Dataset contributions**: Additional species or regional data
2. **Model improvements**: Architecture enhancements or training optimizations  
3. **Application development**: Downstream task implementations
4. **Bug reports**: Issues with reproducibility or performance

### Development Setup

```bash
# Development installation
git clone https://github.com/Maksim3l/SmartHarvest.git
cd SmartHarvest
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Jan Popič** for experimental environment setup
- **University of Maribor** Faculty of Electrical Engineering and Computer Science
- **Agricultural partners** for orchard access and expert validation
- **VGG Image Annotator** team for annotation tools

---

## 📞 Contact

- **Maksim Loknar**: [maksim.loknar@student.um.si](mailto:maksim.loknar@student.um.si)
- **Uroš Mlakar**: [uros.mlakar@um.si](mailto:uros.mlakar@um.si)
- **Project Issues**: [GitHub Issues](https://github.com/Maksim3l/SmartHarvest/issues)

**University of Maribor**  
Faculty of Electrical Engineering and Computer Science  
Koroška cesta 46, SI-2000 Maribor, Slovenia
