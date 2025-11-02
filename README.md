# Pathology Image Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning framework for semantic segmentation of pathology images to detect cancerous regions. Currently focused on colon cancer with extensibility for other cancer types.

## Overview

This project provides a robust pipeline for pathology image segmentation using UNet++ architecture with ResNet101 backbone. The framework is specifically designed to handle challenges in medical image analysis including class imbalance, color semantics preservation, and limited data scenarios.

## Features

- **Advanced Architecture**: UNet++ with ResNet101 encoder for precise segmentation
- **Medical Image Optimized**: Conservative augmentations that preserve diagnostic color semantics
- **Class Imbalance Handling**: Combined Dice+Focal loss with optimal threshold tuning
- **Training Stability**: Gradient accumulation, early stopping, and comprehensive monitoring
- **Multi-Dataset Ready**: Support for both public and private pathology datasets
- **Extensible Design**: Easy adaptation to different cancer types and datasets


## 📊 Supported Datasets

- **Current**: Cacahis (Public colon cancer dataset)
- **Planned**: Extension to private datasets and other cancer types

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/Pathology-Image-Segmentation.git
cd Pathology-Image-Segmentation

# Install dependencies
pip install -r requirements.txt
```

### Training

``` bash
python scripts/train.py \
    --config configs/config.yaml \
    --data_path /path/to/dataset \
    --output_dir outputs/experiment_1

```

### Evaluation

``` bash
python scripts/evaluate.py \
    --model_path outputs/experiment_1/best_model.pth \
    --data_path /path/to/test_data \
    --output_dir evaluation_results

```


### Configuration
The project uses YAML configuration files for reproducible experimentation:

