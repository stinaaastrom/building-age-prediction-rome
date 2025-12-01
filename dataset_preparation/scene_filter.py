"""
Scene parsing filter using semantic segmentation to identify building-dominant images.
Uses a model trained on ADE20K dataset (150 categories).
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

class SceneFilter:
    """Filter images based on building prominence using scene parsing."""
    
    def __init__(self):
        """Initialize the scene parsing model trained on ADE20K."""
        # Using SegFormer trained on ADE20K (150 categories)
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()
        
        # ADE20K building category ID (category 1 is building)
        # ADE20K categories: 0=wall, 1=building, 2=sky, 3=floor, etc.
        self.building_id = 1
        
    def get_building_ratio(self, image):
        """
        Calculate the proportion of the image occupied by buildings.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            tuple: (building_ratio, is_largest_category)
        """
        # Convert numpy array to PIL if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Prepare image for model
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Get segmentation map
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Upsample to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False
        )
        
        # Get predicted class for each pixel
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
        
        # Calculate statistics
        total_pixels = pred_seg.size
        
        # Count pixels per category
        unique, counts = np.unique(pred_seg, return_counts=True)
        category_counts = dict(zip(unique, counts))
        
        # Building statistics
        building_pixels = category_counts.get(self.building_id, 0)
        building_ratio = building_pixels / total_pixels
        
        # Check if building is the largest category
        largest_category_id = max(category_counts.items(), key=lambda x: x[1])[0]
        is_largest = (largest_category_id == self.building_id)
        
        return building_ratio, is_largest
    
    def should_keep_image(self, image, min_building_ratio=0.4):
        """
        Determine if image should be kept based on building prominence.
        
        Criteria:
        1) Buildings are the biggest area of the image
        2) Buildings occupy more than min_building_ratio (default 40%) of the image
        
        Args:
            image: PIL Image or numpy array
            min_building_ratio: Minimum proportion of image that must be building (0.4 = 40%)
            
        Returns:
            tuple: (should_keep, building_ratio, is_largest)
        """
        building_ratio, is_largest = self.get_building_ratio(image)
        
        should_keep = is_largest and (building_ratio >= min_building_ratio)
        
        return should_keep, building_ratio, is_largest
    
    def filter_dataset(self, dataset, min_building_ratio=0.4, verbose=True, visualize_rejected=0):
        """
        Filter a dataset to keep only building-dominant images.
        
        Args:
            dataset: HuggingFace dataset with 'image' or 'Picture' field
            min_building_ratio: Minimum building proportion (default 0.4)
            verbose: Print progress information
            visualize_rejected: Number of rejected images to visualize (default 0)
            
        Returns:
            Filtered dataset
        """
        kept_indices = []
        rejected_samples = []  # Store rejected images with reasons
        stats = {
            'total': len(dataset),
            'kept': 0,
            'rejected_not_largest': 0,
            'rejected_too_small': 0
        }
        
        if verbose:
            print(f"Filtering {stats['total']} images based on building prominence...")
            print(f"Criteria: Buildings must be largest category AND occupy >{min_building_ratio*100}% of image")
        
        for idx in range(len(dataset)):
            item = dataset[idx]
            image = item.get('Picture') or item.get('image')
            
            if image is None:
                continue
            
            should_keep, building_ratio, is_largest = self.should_keep_image(
                image, min_building_ratio
            )
            
            if should_keep:
                kept_indices.append(idx)
                stats['kept'] += 1
            else:
                # Store rejected samples for visualization
                if len(rejected_samples) < visualize_rejected:
                    reason = "Building not largest category" if not is_largest else f"Building only {building_ratio*100:.1f}% (< {min_building_ratio*100}%)"
                    rejected_samples.append({
                        'image': image,
                        'building_ratio': building_ratio,
                        'is_largest': is_largest,
                        'reason': reason,
                        'name': item.get('Building', 'Unknown'),
                        'idx': idx
                    })
                
                if not is_largest:
                    stats['rejected_not_largest'] += 1
                else:
                    stats['rejected_too_small'] += 1
            
            # Progress update
            if verbose and (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{stats['total']} images... "
                      f"Kept: {stats['kept']} ({stats['kept']/(idx+1)*100:.1f}%)")
        
        if verbose:
            print(f"\nFiltering complete:")
            print(f"  Total images: {stats['total']}")
            print(f"  Kept: {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
            print(f"  Rejected (building not largest): {stats['rejected_not_largest']}")
            print(f"  Rejected (building < {min_building_ratio*100}%): {stats['rejected_too_small']}")
        
        # Visualize rejected samples if requested
        if visualize_rejected > 0 and len(rejected_samples) > 0:
            self._visualize_rejected_samples(rejected_samples)
        
        # Return filtered dataset
        filtered_dataset = dataset.select(kept_indices)
        return filtered_dataset
    
    def _visualize_rejected_samples(self, rejected_samples):
        """Visualize rejected images with segmentation maps"""
        import matplotlib.pyplot as plt
        
        n_samples = len(rejected_samples)
        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 5 * n_samples))
        
        if n_samples == 1:
            axes = [axes]
        
        print(f"\n--- Visualizing {n_samples} Rejected Images ---")
        
        for i, sample in enumerate(rejected_samples):
            image = sample['image']
            
            # Original image
            axes[i][0].imshow(image)
            axes[i][0].set_title(f"Original: {sample['name']}\n{sample['reason']}")
            axes[i][0].axis('off')
            
            # Get segmentation
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False
            )
            
            pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            
            # Segmentation map
            im = axes[i][1].imshow(pred_seg, cmap='tab20')
            axes[i][1].set_title(f"Segmentation\nBuilding: {sample['building_ratio']*100:.1f}% | Largest: {sample['is_largest']}")
            axes[i][1].axis('off')
            plt.colorbar(im, ax=axes[i][1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save visualization
        save_path = "rejected_images_visualization.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved rejected images visualization to {save_path}")
        plt.show()
        plt.close()


def visualize_segmentation(image, scene_filter, save_path=None):
    """
    Visualize the segmentation map for an image.
    
    Args:
        image: PIL Image or numpy array
        scene_filter: SceneFilter instance
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    # Convert numpy array to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Get segmentation
    inputs = scene_filter.processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = scene_filter.model(**inputs)
        logits = outputs.logits
    
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    # Calculate statistics
    building_ratio, is_largest = scene_filter.get_building_ratio(image)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Segmentation map
    # Highlight buildings in a distinct color
    seg_colored = pred_seg.copy()
    im = axes[1].imshow(seg_colored, cmap='tab20')
    axes[1].set_title(f"Segmentation Map\nBuilding: {building_ratio*100:.1f}% | "
                     f"Largest: {is_largest}")
    axes[1].axis('off')
    
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test the scene filter
    from datasets import load_dataset
    
    print("Loading scene filter...")
    scene_filter = SceneFilter()
    
    # Test on a sample image
    print("\nTesting on sample dataset...")
    dataset = load_dataset("parquet", 
                          data_files="year_guessr_data/csv/train.csv",
                          split="train[:10]")
    
    # Test first image
    if len(dataset) > 0:
        item = dataset[0]
        image = item.get('Picture') or item.get('image')
        
        if image:
            print("\nAnalyzing first image...")
            should_keep, building_ratio, is_largest = scene_filter.should_keep_image(image)
            
            print(f"Building ratio: {building_ratio*100:.2f}%")
            print(f"Is largest category: {is_largest}")
            print(f"Should keep: {should_keep}")
            
            # Visualize
            visualize_segmentation(image, scene_filter)
