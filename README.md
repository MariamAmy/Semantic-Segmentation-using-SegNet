# Semantic Segmentation using SegNet

## Overview
This project implements **SegNet**, a deep convolutional encoder-decoder network designed for semantic pixel-wise segmentation. Semantic segmentation classifies each pixel in an image into predefined categories, providing detailed object boundaries compared to object detection techniques.

## Project Description
This project implements the **SegNet** architecture for semantic segmentation using the **VOC2012** dataset. SegNet uses a convolutional encoder-decoder structure based on the **VGG16** architecture, with modifications for pixel-wise classification.

### Key Features:
- **Encoder**: Replicates the 13 convolutional layers of VGG16 without fully connected layers for efficiency.
- **Decoder**: Uses pooling indices from the encoder for upsampling without learning additional weights.
- **Pixel-wise Classification**: Final layer performs pixel-wise labeling for semantic segmentation tasks.

## Dataset: VOC2012
The **PASCAL VOC2012** dataset is used for training and validation. It includes:
- Images with pixel-wise annotations for 20 object classes.
- Bounding boxes and object labels for each image.

**Dataset Link:** [PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)

## SegNet Architecture
- **Encoder**: Replicates the 13 convolutional layers of VGG16, excluding fully connected layers.
- **Decoder**: Upsamples feature maps using the max-pooling indices from the encoder.
- **Final Layer**: Performs pixel-wise classification to assign each pixel a class label.

## Applying CUDA
To improve performance and reduce training time:
- **CUDA (Compute Unified Device Architecture)** is used for parallel computation.
- The implementation is optimized for both memory efficiency and speed, making it suitable for real-time applications.

## Requirements
- Python 3.x
- TensorFlow / PyTorch
- CUDA-enabled GPU
- VOC2012 Dataset
