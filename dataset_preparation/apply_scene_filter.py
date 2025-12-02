"""
Apply scene parsing filter to the Rome building dataset.
This will filter images to keep only those where:
1) Buildings are the biggest area of the image
2) Buildings occupy more than 40% of the image
"""

from dataset_preparation.filter_italy_dataset import ItalyDataset
from dataset_preparation.scene_filter import SceneFilter
from pathlib import Path
import sys

def apply_scene_filter_to_datasets(min_building_ratio=0.4):
    """
    Apply scene parsing filter to train, test, and validation datasets.
    
    Args:
        min_building_ratio: Minimum proportion of image that must be building (default 0.4 = 40%)
    """
    # Initialize Dataset Handler
    italy_geojson_path = Path.cwd() / 'resources' / 'italy_borders.geojson'
    dataset_name = "Morris0401/Year-Guessr-Dataset"
    italy_data = ItalyDataset(italy_geojson_path, dataset_name)
    
    # Initialize Scene Filter
    print("Loading scene parsing model (SegFormer trained on ADE20K)...")
    scene_filter = SceneFilter()
    print("Model loaded successfully!\n")
    
    # Process each split
    splits = ["train", "test", "valid"]
    filtered_datasets = {}
    
    for split in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} dataset")
        print(f"{'='*60}")
        
        # Load filtered dataset (already filtered by geography)
        dataset = italy_data.get_filtered_dataset(split=split)
        print(f"Loaded {len(dataset)} images from {split} split (after geographic filtering)")
        
        if len(dataset) == 0:
            print(f"No data in {split} split, skipping...")
            filtered_datasets[split] = dataset
            continue
        
        # Apply scene parsing filter
        print(f"\nApplying scene parsing filter...")
        filtered_dataset = scene_filter.filter_dataset(
            dataset, 
            min_building_ratio=min_building_ratio,
            verbose=True
        )
        
        filtered_datasets[split] = filtered_dataset
        
        print(f"\n{split.upper()} summary:")
        print(f"  Before scene filter: {len(dataset)} images")
        print(f"  After scene filter: {len(filtered_dataset)} images")
        print(f"  Retention rate: {len(filtered_dataset)/len(dataset)*100:.1f}%")
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    for split in splits:
        original = italy_data.get_filtered_dataset(split=split)
        filtered = filtered_datasets[split]
        print(f"{split.upper():>10}: {len(original):>5} → {len(filtered):>5} images "
              f"({len(filtered)/len(original)*100:.1f}% retained)")
    
    return filtered_datasets


def test_scene_filter_on_samples(num_samples=5):
    """
    Test the scene filter on a few sample images and visualize results.
    
    Args:
        num_samples: Number of images to test and visualize
    """
    from dataset_preparation.scene_filter import visualize_segmentation
    import random
    
    # Load dataset
    italy_geojson_path = Path.cwd() / 'resources' / 'italy_borders.geojson'
    dataset_name = "Morris0401/Year-Guessr-Dataset"
    italy_data = ItalyDataset(italy_geojson_path, dataset_name)
    
    dataset = italy_data.get_filtered_dataset(split="train")
    
    if len(dataset) == 0:
        print("No data found!")
        return
    
    # Initialize Scene Filter
    print("Loading scene parsing model...")
    scene_filter = SceneFilter()
    
    # Select random samples
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    print(f"\nTesting on {len(indices)} random images...\n")
    
    for i, idx in enumerate(indices):
        item = dataset[idx]
        image = item.get('Picture') or item.get('image')
        building_name = item.get('Building', 'Unknown')
        year = item.get('Year', 'Unknown')
        
        if image is None:
            continue
        
        print(f"\nImage {i+1}/{len(indices)}: {building_name} ({year})")
        
        should_keep, building_ratio, is_largest = scene_filter.should_keep_image(image)
        
        print(f"  Building ratio: {building_ratio*100:.1f}%")
        print(f"  Is largest category: {is_largest}")
        print(f"  Decision: {'✓ KEEP' if should_keep else '✗ REJECT'}")
        
        # Visualize
        visualize_segmentation(
            image, 
            scene_filter,
            save_path=f"scene_filter_test_{i+1}.png"
        )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply scene parsing filter to building dataset")
    parser.add_argument("--mode", choices=["test", "filter"], default="test",
                       help="Mode: 'test' to visualize samples, 'filter' to process all data")
    parser.add_argument("--samples", type=int, default=5,
                       help="Number of samples to visualize in test mode")
    parser.add_argument("--min-building-ratio", type=float, default=0.4,
                       help="Minimum building ratio (0-1, default 0.4)")
    
    args = parser.parse_args()
    
    if args.mode == "test":
        print("Running in TEST mode - visualizing sample images...")
        test_scene_filter_on_samples(num_samples=args.samples)
    else:
        print("Running in FILTER mode - processing all datasets...")
        confirm = input(f"This will filter datasets with min_building_ratio={args.min_building_ratio}. Continue? (y/n): ")
        if confirm.lower() == 'y':
            filtered_datasets = apply_scene_filter_to_datasets(
                min_building_ratio=args.min_building_ratio
            )
            print("\nFiltering complete! You can now integrate these filtered datasets into your training pipeline.")
        else:
            print("Cancelled.")
