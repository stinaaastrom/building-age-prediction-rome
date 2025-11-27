import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import cv2
import random
import torch
from torchvision import models
import torchvision.transforms as T



class ImageProcessing:

    def __init__(self, use_augmentation: bool = True):
        """Image preprocessing wrapper.

        Args:
            use_augmentation: If False, only deterministic resizing is applied.
        """
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.augment = T.Compose([
                T.Resize((256, 256)),
                T.RandomHorizontalFlip(0.5),
                T.RandomRotation(5),
                T.ColorJitter(brightness=0.15, contrast=0.15),
                T.Resize((224, 224)),
            ])
        else:
            # Deterministic path: just resize to model input size
            self.augment = T.Compose([
                T.Resize((224, 224)),
            ])
        self.preprocess = models.ResNet50_Weights.DEFAULT.transforms()

        
    def compute_sobel(self, img):
        """
        Compute Sobel edge detection and return as single channel.
        Input: PIL Image
        Output: numpy array (H, W) normalized to 0-1
        """
        # Convert to grayscale and add light Gaussian blur before Sobel
        img_np = cv2.GaussianBlur(np.array(img.convert('L')), (3,3), 0)

        
        # Compute Sobel gradients
        sobelx = cv2.Sobel(img_np, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img_np, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute magnitude
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize to 0-1
        sobel_magnitude = (sobel_magnitude - sobel_magnitude.min()) / (sobel_magnitude.max() - sobel_magnitude.min() + 1e-8)
        
        return sobel_magnitude

    def compute_sobel_clahe(self, img, clip_limit=2.0, tile_grid_size=(8,8)):
        """
        Compute Sobel magnitude then apply CLAHE locally to enhance edge contrast.
        Returns float array in range 0-1.
        """
        # Grayscale + slight blur to reduce noise amplification
        gray = cv2.GaussianBlur(np.array(img.convert('L')), (3,3), 0)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        # Normalize for CLAHE input (0-255 uint8)
        mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        mag_uint8 = (mag_norm * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        mag_clahe = clahe.apply(mag_uint8)
        # Back to float 0-1
        return mag_clahe.astype(np.float32) / 255.0

    def compute_laplacian(self, img, ksize: int = 3):
        """Compute Laplacian focus/edge channel and normalize to 0-1."""
        gray = cv2.GaussianBlur(np.array(img.convert('L')), (3,3), 0)
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        lap_abs = np.abs(lap)
        lap_norm = (lap_abs - lap_abs.min()) / (lap_abs.max() - lap_abs.min() + 1e-8)
        return lap_norm.astype(np.float32)
    
    def image_processing(self, img):
        """
        Process image with augmentation and create 4-channel output (RGB + Sobel).
        Input: PIL Image
        Output: 4-channel tensor (C, H, W) where C=4
        """
        
        img_np = np.array(img)
        img_yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        img = Image.fromarray(img_eq)

        
        # Apply augmentation or plain resize depending on configuration
        img_aug = self.augment(img)
        
        # Compute Sobel (CLAHE) and Laplacian channels
        sobel_channel = self.compute_sobel_clahe(img_aug)
        lap_channel = self.compute_laplacian(img_aug)
        
        # Apply ResNet preprocessing to RGB (returns 3-channel tensor)
        img_tensor = self.preprocess(img_aug)
        
        # Resize Sobel to match tensor dimensions
        # img_tensor shape: (3, 224, 224)
        # Resize edge channels to match tensor dimensions
        sobel_resized = cv2.resize(sobel_channel, (img_tensor.shape[2], img_tensor.shape[1]))
        lap_resized = cv2.resize(lap_channel, (img_tensor.shape[2], img_tensor.shape[1]))
        sobel_tensor = torch.from_numpy(sobel_resized).float().unsqueeze(0)
        lap_tensor = torch.from_numpy(lap_resized).float().unsqueeze(0)

        # Normalize edge channels (simple center-scale)
        sobel_tensor = (sobel_tensor - 0.5) / 0.25
        lap_tensor = (lap_tensor - 0.5) / 0.25

        # Concatenate RGB + Sobel + Laplacian -> 5 channels
        img_5channel = torch.cat([img_tensor, sobel_tensor, lap_tensor], dim=0)
        return img_5channel
  