"""
Scene parsing filter using semantic segmentation to identify building facade images.
Filters out interior shots and construction sites.
Uses a model trained on ADE20K dataset (150 categories).
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

class SceneFilter:
    """Filter images to keep only exterior building facades, excluding interiors and construction sites."""
    
    def __init__(self):
        """Initialize the scene parsing model trained on ADE20K."""
        # Using SegFormer trained on ADE20K (150 categories)
        model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()
        
        # ADE20K category IDs
        # Key categories: 0=wall, 1=building, 2=sky, 3=floor, 4=tree, 5=ceiling, etc.
        self.building_id = 1
        self.sky_id = 2
        
        # Interior indicators (if these dominate, it's likely an interior shot)
        self.interior_ids = [0, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25]
        # 0=wall, 3=floor, 5=ceiling, 6=bed, 7=windowpane, 8=grass, 9=cabinet, 10=sidewalk, 
        # 11=person, 13=earth/ground, 14=door, 15=table, 16=mountain, 18=plant, 19=curtain,
        # 22=shelf, 23=stairs, 24=escalator, 25=ottoman
        
    def analyze_scene(self, image):
        """
        Analyze the scene to determine if it's a building facade.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            dict: Scene analysis with ratios for building, sky, interior, construction
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
        
        # Calculate ratios
        building_ratio = category_counts.get(self.building_id, 0) / total_pixels
        sky_ratio = category_counts.get(self.sky_id, 0) / total_pixels
        
        # Interior ratio (sum of all interior categories)
        interior_pixels = sum(category_counts.get(cat_id, 0) for cat_id in self.interior_ids)
        interior_ratio = interior_pixels / total_pixels
        
        # Check if building is the largest category
        largest_category_id = max(category_counts.items(), key=lambda x: x[1])[0]
        is_building_largest = (largest_category_id == self.building_id)
        
        return {
            'building_ratio': building_ratio,
            'sky_ratio': sky_ratio,
            'interior_ratio': interior_ratio,
            'is_building_largest': is_building_largest,
            'pred_seg': pred_seg
        }
    
    def should_keep_image(self, image, min_building_ratio=0.15, min_sky_ratio=0.05, max_interior_ratio=0.4):
        """
        Determine if image is an exterior building facade (not interior).
        
        Criteria for keeping (more permissive for distant buildings):
        1) Building visible (>15% of image)
        2) Sky visible (>5%) - indicates exterior shot
        3) Interior elements <40% - not an interior shot
        
        Note: Building doesn't need to be largest if sky is visible (allows distant buildings)
        
        Args:
            image: PIL Image or numpy array
            min_building_ratio: Minimum building proportion (default 0.15)
            min_sky_ratio: Minimum sky proportion (default 0.05)
            max_interior_ratio: Maximum interior elements (default 0.4)
            
        Returns:
            tuple: (should_keep, analysis_dict, rejection_reason)
        """
        analysis = self.analyze_scene(image)
        
        # More flexible criteria: building visible AND sky present (exterior) AND low interior
        # Building doesn't need to be largest - allows for distant buildings with lots of sky
        has_building = analysis['building_ratio'] >= min_building_ratio
        has_sky = analysis['sky_ratio'] >= min_sky_ratio
        not_interior = analysis['interior_ratio'] < max_interior_ratio
        
        is_facade = has_building and has_sky and not_interior
        
        # Determine rejection reason if not kept
        rejection_reason = None
        if not is_facade:
            if analysis['building_ratio'] < min_building_ratio:
                rejection_reason = f"Building only {analysis['building_ratio']*100:.1f}% (need ≥{min_building_ratio*100}%)"
            elif analysis['sky_ratio'] < min_sky_ratio:
                rejection_reason = f"No sky visible ({analysis['sky_ratio']*100:.1f}% < {min_sky_ratio*100}%) - likely interior"
            elif analysis['interior_ratio'] >= max_interior_ratio:
                rejection_reason = f"Interior elements {analysis['interior_ratio']*100:.1f}% (≥{max_interior_ratio*100}%) - interior shot"
        
        return is_facade, analysis, rejection_reason
    
    def filter_dataset(self, dataset, min_building_ratio=0.15, min_sky_ratio=0.05, max_interior_ratio=0.4, verbose=True, visualize_rejected=0):
        """
        Filter a dataset to keep only exterior building facade images.
        More permissive to include distant buildings.
        
        Args:
            dataset: HuggingFace dataset with 'image' or 'Picture' field
            min_building_ratio: Minimum building proportion (default 0.15)
            min_sky_ratio: Minimum sky proportion for exterior (default 0.05)
            max_interior_ratio: Maximum interior elements (default 0.4)
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
            'rejected_interior': 0,
            'rejected_no_sky': 0,
            'rejected_other': 0
        }
        
        if verbose:
            print(f"Filtering {stats['total']} images to keep only building facades...")
            print(f"Criteria: Exterior building facades (includes distant buildings)")
            print(f"  - Building visible ≥{min_building_ratio*100}%")
            print(f"  - Sky visible ≥{min_sky_ratio*100}% (exterior indicator)")
            print(f"  - Interior elements <{max_interior_ratio*100}%")
        
        for idx in range(len(dataset)):
            item = dataset[idx]
            image = item.get('Picture') or item.get('image')
            
            if image is None:
                continue
            
            is_facade, analysis, rejection_reason = self.should_keep_image(
                image, min_building_ratio, min_sky_ratio, max_interior_ratio
            )
            
            if is_facade:
                kept_indices.append(idx)
                stats['kept'] += 1
            else:
                # Store rejected samples for visualization
                if len(rejected_samples) < visualize_rejected:
                    rejected_samples.append({
                        'image': image,
                        'analysis': analysis,
                        'reason': rejection_reason,
                        'name': item.get('Building', 'Unknown'),
                        'idx': idx
                    })
                
                # Categorize rejection
                if "interior" in rejection_reason.lower():
                    stats['rejected_interior'] += 1
                elif "sky" in rejection_reason.lower():
                    stats['rejected_no_sky'] += 1
                else:
                    stats['rejected_other'] += 1
            
            # Progress update
            if verbose and (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{stats['total']} images... "
                      f"Kept: {stats['kept']} ({stats['kept']/(idx+1)*100:.1f}%)")
        
        if verbose:
            print(f"\nFiltering complete:")
            print(f"  Total images: {stats['total']}")
            print(f"  Kept (facades): {stats['kept']} ({stats['kept']/stats['total']*100:.1f}%)")
            print(f"  Rejected (interior): {stats['rejected_interior']}")
            print(f"  Rejected (no sky/likely interior): {stats['rejected_no_sky']}")
            print(f"  Rejected (other): {stats['rejected_other']}")
        
        # Visualize rejected samples if requested
        if visualize_rejected > 0 and len(rejected_samples) > 0:
            self._visualize_rejected_samples(rejected_samples)
        
        # Return filtered dataset
        filtered_dataset = dataset.select(kept_indices)
        return filtered_dataset
    
    def _visualize_rejected_samples(self, rejected_samples):
        """Visualize rejected images with facade analysis"""
        import matplotlib.pyplot as plt
        
        n_samples = len(rejected_samples)
        fig, axes = plt.subplots(n_samples, 2, figsize=(14, 5 * n_samples))
        
        if n_samples == 1:
            axes = [axes]
        
        print(f"\n--- Visualizing {n_samples} Rejected Images ---")
        
        for i, sample in enumerate(rejected_samples):
            image = sample['image']
            analysis = sample['analysis']
            
            # Original image
            axes[i][0].imshow(image)
            title = f"Original: {sample['name']}\n{sample['reason']}"
            axes[i][0].set_title(title, fontsize=10)
            axes[i][0].axis('off')
            
            # Segmentation map with analysis
            pred_seg = analysis['pred_seg']
            im = axes[i][1].imshow(pred_seg, cmap='tab20')
            
            stats_text = (
                f"Building: {analysis['building_ratio']*100:.1f}%\n"
                f"Sky: {analysis['sky_ratio']*100:.1f}%\n"
                f"Interior: {analysis['interior_ratio']*100:.1f}%"
            )
            axes[i][1].set_title(f"Segmentation Analysis\n{stats_text}", fontsize=9)
            axes[i][1].axis('off')
            plt.colorbar(im, ax=axes[i][1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # Save visualization
        save_path = "rejected_facades_visualization.png"
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