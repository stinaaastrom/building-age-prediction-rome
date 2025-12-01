"""
Image preprocessing module for building age prediction models.
Provides processing pipelines for both PyTorch (SVR) and TensorFlow/Keras (CNN) models.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import cv2
import random
import torch
from torchvision import models
import torchvision.transforms as T

# Probability that augmentation is applied during training
AUGMENTATION_P = 0.7
IMAGE_SIZE = 224

class ImageProcessing:
    """
    Image preprocessing class for building age prediction.
    Supports both PyTorch (channel-first) and Keras/TensorFlow (channel-last) formats.
    """

    def __init__(self):
        """Initialize image processing pipelines and transformations."""
        self.resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE))
        self.preprocess = models.ResNet50_Weights.DEFAULT.transforms()

        # Data augmentation pipeline for training robustness
        self.augmentations = T.Compose([
                # Random crop provides scale invariance and robustness
                T.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                # Horizontal flip for left-right invariance
                T.RandomHorizontalFlip(p=0.5),
                # Slight rotation to handle camera angles
                T.RandomApply([T.RandomRotation(degrees=10)], p=0.5),
                # Small translations and shear for position invariance
                T.RandomApply([T.RandomAffine(degrees=0, translate=(0.05,0.05), shear=5)], p=0.3),
                # Perspective transform makes model robust to camera viewpoints
                T.RandomApply([T.RandomPerspective(distortion_scale=0.2, p=1.0)], p=0.3),
                # Moderate color jittering for lighting condition robustness
                T.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05),
                # Occasional blur to handle image quality variations
                T.RandomApply([T.GaussianBlur(kernel_size=(3,3), sigma=(0.1,1.0))], p=0.15),
            ])

    def image_processing(self, img, use_augmentation=False):
        """
        Process images for PyTorch models (ResNet50 + SVR).
        
        Returns images in channel-first format (C, H, W) as PyTorch tensors.
        """
        # Ensure image is in RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Apply augmentation with probability AUGMENTATION_P during training
        if use_augmentation:
            if random.random() < AUGMENTATION_P:
                img = self.augmentations(img)

        # Resize to standard size
        img_resized = self.resize(img)
        
        # Apply ResNet preprocessing (normalization, tensor conversion to channel-first)
        img_tensor = self.preprocess(img_resized)

        return img_tensor
    
    def image_processing_for_cnn(self, img, use_augmentation=False):
        """
        Process images for Keras/TensorFlow CNN models (DenseNet121).
        
        Returns images in channel-last format (H, W, C) as numpy arrays.
        This is required for Keras/TensorFlow which expects (batch, height, width, channels).
        """
        # Ensure image is in RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Apply augmentation with probability AUGMENTATION_P during training
        if use_augmentation:
            if random.random() < AUGMENTATION_P:
                img = self.augmentations(img)
        
        # Resize to standard size (224x224)
        img_resized = self.resize(img)
        
        # Convert PIL Image to numpy array and scale to [0, 1]
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        
        # Apply ImageNet normalization (same as PyTorch preprocessing)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_normalized = (img_array - mean) / std
        
        # Return in channel-last format (H, W, C) for Keras/TensorFlow
        return img_normalized
