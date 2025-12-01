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

    def __init__(self):
        self.resize = T.Resize((224, 224))
        self.preprocess = models.ResNet50_Weights.DEFAULT.transforms()

    def apply_clahe_to_rgb(self, img):
        """
        Apply CLAHE only to the Luminance channel (Lab color space).
        Keeps colors natural while enhancing textures.
        """
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        lab_clahe = cv2.merge((l_clahe, a, b))
        rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

        return Image.fromarray(rgb_clahe)

    def compute_sobel_edge_mask(self, img):
        """Return normalized Sobel magnitude mask (0–1)."""
        gray = cv2.GaussianBlur(np.array(img.convert('L')), (3, 3), 0)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)

        mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
        return mag_norm

    def enhance_with_edges(self, img):
        """
        Apply texture/edge enhancement to RGB using Sobel mask.
        Soft sharpening based on Sobel magnitude.
        """
        sobel = self.compute_sobel_edge_mask(img)
        sobel = np.expand_dims(sobel, axis=2)  # shape (H, W, 1)

        rgb = np.array(img).astype(np.float32) / 255.0

        # Weighted blending: 20% sobel, 80% original
        enhanced = rgb * 0.9 + sobel * 0.1
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)

        return Image.fromarray(enhanced)

    def image_processing(self, img):
        """
        Image processing steps:
        1. CLAHE enhancement
        2. Light Sobel sharpening
        3. Resize
        4. ResNet preprocessing
        """
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Step 1: CLAHE (primary enhancement)
        # img = self.apply_clahe_to_rgb(img)

        # Step 2: light Sobel sharpening (secondary enhancement)
        # img = self.enhance_with_edges(img)

        # Step 3: Resize to 224x224
        img_resized = self.resize(img)

        # Step 4: Preprocessing for ResNet (normalization, tensor conversion)
        img_tensor = self.preprocess(img_resized)

        return img_tensor
