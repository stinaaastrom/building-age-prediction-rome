from dataset_preparation.filter_italy_dataset import ItalyDataset
from dataset_preparation.scene_filter import SceneFilter
from model_training.train_svr_model import SVRModel
from model_training.train_cnn_model import CNNModel
from model_training.train_gradient_boosting_model import GradientBoostingModel
from result_visualization.visualize_predictions import PredictionVisualizer
from result_visualization.visualize_feature_space import FeatureSpaceVisualizer
from result_visualization.visualize_errors_geographically import GeographicErrorVisualizer
from result_visualization.find_worst_predictions import WorstPredictionsFinder
from result_visualization.confusion_matrix_age import AgeConfusionMatrix
from result_visualization.visualize_period_distribution import PeriodDistributionVisualizer
from pathlib import Path
from typing import Literal
import pickle
import random
from datasets import concatenate_datasets

CACHE_DIR = Path.cwd() / 'dataset_preparation' / 'cache'

def main():
    train_dataset = provide_dataset('train', use_cache=True, model_type='svr')

    # Visualize Period Distribution
    """ print("\n--- Visualizing Period Distribution ---")
    period_visualizer = PeriodDistributionVisualizer()
    period_visualizer.visualize(train_dataset) """

    model = train_model('svr', train_dataset, use_cache=False)
    
    model.evaluate(provide_dataset('test', use_cache=True))
    
    """ # Visualize predictions
    visualizer = PredictionVisualizer(model, model_type=method)
    visualizer.visualize(test_dataset, num_samples=3) """

    # Confusion Matrix by Age Period
    print("\n--- Generating Confusion Matrix ---")
    cm_analyzer = AgeConfusionMatrix(model, model_type='svr')
    cm_analyzer.compute_confusion_matrix(provide_dataset('test'))
"""     cm_analyzer.analyze_errors_by_period(provide_dataset('test')) """

"""     # Visualize Geographic Error Distribution
    print("\n--- Visualizing Geographic Error Distribution ---")
    geo_error_visualizer = GeographicErrorVisualizer(model)
    geo_error_visualizer.visualize_errors_on_map(provide_dataset('test'))
    geo_error_visualizer.visualize_error_density_regions(provide_dataset('test')) """

"""     # Additional visualizations (SVR only)
    # Visualize Feature Space
    print("\n--- Visualizing Feature Space ---")
    feature_visualizer = FeatureSpaceVisualizer(model)
    feature_visualizer.visualize(provide_dataset('train'))

    # Find Worst Predictions
    print("\n--- Finding Worst Predictions ---")
    worst_finder = WorstPredictionsFinder(model)
    worst_finder.find_worst(provide_dataset('test')) """

def provide_dataset(dataset_type: Literal['train','test','valid'], use_cache: bool = True, model_type: str = None):
    print("\n--- Loading, Filtering by Geography and Applying Facade Detection Filter ---")
    # Check for cached dataset
    cache_path = CACHE_DIR / f'{dataset_type}_dataset.pkl'
    
    if use_cache and cache_path.exists():
        print(f"\n--- Loading cached {dataset_type} dataset from {cache_path} ---")
        with open(cache_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset
    
    # Initialize Dataset Handler
    italy_geojson_path = Path.cwd() / 'resources' / 'italy_borders.geojson'
    dataset_name = "Morris0401/Year-Guessr-Dataset"
    italy_data = ItalyDataset(italy_geojson_path, dataset_name)

    print("Loading scene parsing model (SegFormer trained on ADE20K)...")
    scene_filter = SceneFilter()

    print("Filtering datasets to keep only exterior building facades:")
    print("  - Excludes interior shots (walls, floors, ceilings dominant)")
    print("  - Requires visible sky (exterior indicator)")

    dataset = italy_data.get_filtered_dataset(split=dataset_type)
    dataset = scene_filter.filter_dataset(dataset)

    # Balance dataset by period
    print("\n--- Balancing dataset by period ---")
    period_visualizer = PeriodDistributionVisualizer()
    
    # Augment underrepresented periods with validation data (only for SVR training)
    if dataset_type == 'train' and model_type == 'svr':
        print("\n--- Augmenting underrepresented periods with validation data ---")
        # Calculate current distribution
        period_counts = {}
        years = dataset['Year']
        for year in years:
            p = period_visualizer._get_period(year)
            if p is not None:
                period_counts[p] = period_counts.get(p, 0) + 1
        
        if period_counts:
            max_count = max(period_counts.values())
            
            print("Loading validation dataset for augmentation...")
            # Recursive call to get validation dataset (can use cache)
            valid_dataset = provide_dataset('valid', use_cache=True)
            
            indices_to_add = []
            valid_years = valid_dataset['Year']
            for idx, year in enumerate(valid_years):
                p = period_visualizer._get_period(year)
                if p is not None:
                    # Add sample if it belongs to an underrepresented period
                    if period_counts.get(p, 0) < max_count:
                        indices_to_add.append(idx)
            
            if indices_to_add:
                print(f"Adding {len(indices_to_add)} samples from validation dataset.")
                augment_data = valid_dataset.select(indices_to_add)
                dataset = concatenate_datasets([dataset, augment_data])
            else:
                print("No suitable samples found in validation dataset to augment underrepresented periods.")

    period_indices = {}
    
    years = dataset['Year']
    for idx, year in enumerate(years):
        period = period_visualizer._get_period(year)
        if period is not None:
            if period not in period_indices:
                period_indices[period] = []
            period_indices[period].append(idx)
            
    if period_indices:
        min_count = min(len(indices) for indices in period_indices.values())
        print(f"Balancing to {min_count} images per period.")
        
        balanced_indices = []
        for period, indices in period_indices.items():
            selected_indices = random.sample(indices, min_count)
            balanced_indices.extend(selected_indices)
            
        balanced_indices.sort()
        dataset = dataset.select(balanced_indices)
        print(f"Balanced dataset size: {len(dataset)}")
    else:
        print("Warning: No valid periods found in dataset.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Saving {dataset_type} dataset to cache: {cache_path} ---")
    with open(cache_path, 'wb') as f:
        pickle.dump(dataset, f)

    return dataset
    

def train_model(training_method: Literal['svr', 'cnn', 'gbm'], train_dataset, use_cache: bool = True):
    print("\n--- Starting Training ---")
    
    match training_method:
        case 'svr':
            model = SVRModel()
            model_path = Path.cwd() / 'model_training' / 'svm.joblib'

            if use_cache and model.load_model(model_path):
                print("Skipping training as cached model was loaded.")
            else:
                model.train(train_dataset)
                model.save_model(model_path)
        
        case 'cnn':
            model = CNNModel()
            model_path = Path.cwd() / 'model_training' / 'cnn_age_model.keras'
            
            if use_cache and model.load_model(model_path):
                print("Skipping training as cached model was loaded.")
            else:
                model.train_cnn(train_dataset, val_dataset=provide_dataset('valid', use_cache=True), epochs=20, batch_size=64)
                model.save_model(model_path)
        
        case 'gbm':
            model = GradientBoostingModel()
            model_path = Path.cwd() / 'model_training' / 'gbm.joblib'
            
            if use_cache and model.load_model(model_path):
                print("Skipping training as cached model was loaded.")
            else:
                model.train(train_dataset, val_dataset=provide_dataset('valid', use_cache=True))
                model.save_model(model_path)
    
    return model


if __name__ == "__main__":
    main()
