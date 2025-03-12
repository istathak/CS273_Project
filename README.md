# Image Classification using CNN

## Overview
This project implements an image classification model using a Convolutional Neural Network (CNN) in PyTorch. The goal is to classify images into two categories (binary classification). The dataset consists of grayscale images in `.png` format, and the model is trained to distinguish between different classes effectively.

## Dataset
- Images are in `.png` format.
- Each image has a shape of `1x224x224` (grayscale, height = 224, width = 224).
- The dataset is preprocessed and loaded using a custom PyTorch `DataLoader`.

## Model Architecture
The CNN architecture consists of:
1. **Conv2D Layer 1**: 32 filters, kernel size (3x3), ReLU activation, followed by MaxPooling (2x2).
2. **Conv2D Layer 2**: 64 filters, kernel size (3x3), ReLU activation, followed by MaxPooling (2x2).
3. **Conv2D Layer 3**: 128 filters, kernel size (3x3), ReLU activation, followed by MaxPooling (2x2).
4. **Fully Connected Layer (fc1)**: 128 neurons, followed by a Dropout layer (0.5).
5. **Output Layer (fc2)**: 1 neuron with Sigmoid activation (binary classification).

## Installation
### Prerequisites
Ensure you have Python and the required dependencies installed:
```bash
pip install torch torchvision numpy matplotlib bokeh sklearn
```

