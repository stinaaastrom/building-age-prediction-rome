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
from sklearn.model_selection import KFold, train_test_split
import numpy as np

CACHE_DIR = Path.cwd() / 'dataset_preparation' / 'cache'

def main():
    # K-Fold Cross Validation on full dataset
    print("\n=== Starting K-Fold Cross Validation ===")
    full_dataset = provide_dataset('wiki_dataset', use_cache=True)
    
    model_type = 'gbm' # Options: 'svr', 'cnn', 'gbm'
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    scores = []
    
    # Convert to list of indices for splitting
    indices = list(range(len(full_dataset)))
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(indices)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        
        # Create train/test splits for this fold
        # Split train_idx into train and validation (e.g. 80/20) for models that need it
        train_indices, val_indices = train_test_split(train_idx, test_size=0.2, random_state=42)
        
        train_dataset = full_dataset.select(train_indices)
        val_dataset = full_dataset.select(val_indices)
        test_dataset = full_dataset.select(test_idx)
        
        # Balance training dataset
        # We will just use the downsampling balancing here.
        train_dataset = balance_dataset(train_dataset)
        
        # Train model
        extension = 'keras' if model_type == 'cnn' else 'joblib'
        model_filename = f'{model_type}_fold_{fold+1}.{extension}'
        model = train_model(model_type, train_dataset, val_dataset=val_dataset, use_cache=True, model_filename=model_filename)
        
        # Evaluate
        print(f"Evaluating Fold {fold+1}...")
        metrics = model.evaluate(test_dataset)
        scores.append(metrics)
        
    # Average scores
    print("\n=== K-Fold Cross Validation Results ===")
    avg_mae = np.mean(scores)
    print(f"Average MAE across {k_folds} folds: {avg_mae:.2f} years")
    print(f"Standard Deviation: {np.std(scores):.2f} years")

    # Visualizations for the last fold
    print("\n=== Generating Visualizations for Last Fold ===")
    
    # 1. Period Distribution (on train set of last fold)
    print("\n--- Visualizing Period Distribution (Train) ---")
    period_visualizer = PeriodDistributionVisualizer()
    period_visualizer.visualize(train_dataset)

    # 2. Predictions (on test set)
    print("\n--- Visualizing Predictions ---")
    visualizer = PredictionVisualizer(model, model_type)
    visualizer.visualize(test_dataset, num_samples=3)

    # 3. Confusion Matrix
    print("\n--- Generating Confusion Matrix ---")
    cm_analyzer = AgeConfusionMatrix(model, model_type)
    cm_analyzer.compute_confusion_matrix(test_dataset)
    cm_analyzer.analyze_errors_by_period(test_dataset)

    # 4. Geographic Error
    print("\n--- Visualizing Geographic Error Distribution ---")
    geo_error_visualizer = GeographicErrorVisualizer(model)
    geo_error_visualizer.visualize_errors_on_map(test_dataset)
    geo_error_visualizer.visualize_error_density_regions(test_dataset)

    # 5. Feature Space (on train set)
    print("\n--- Visualizing Feature Space ---")
    feature_visualizer = FeatureSpaceVisualizer(model)
    feature_visualizer.visualize(train_dataset)

    # 6. Worst Predictions
    print("\n--- Finding Worst Predictions ---")
    worst_finder = WorstPredictionsFinder(model)
    worst_finder.find_worst(test_dataset)

def balance_dataset(dataset):
    print("\n--- Balancing dataset by period ---")
    period_visualizer = PeriodDistributionVisualizer()

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
        
    return dataset

def provide_dataset(dataset_type: Literal['train','test','valid','wiki_dataset'], use_cache: bool = True, model_type: str = None):
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

    # Handle wiki_dataset (full dataset)
    dataset = italy_data.get_filtered_dataset(split=dataset_type)
    dataset = scene_filter.filter_dataset(dataset)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Saving {dataset_type} dataset to cache: {cache_path} ---")
    with open(cache_path, 'wb') as f:
        pickle.dump(dataset, f)

    return dataset
    
def train_model(training_method: Literal['svr', 'cnn', 'gbm'], train_dataset, val_dataset=None, use_cache: bool = True, model_filename: str = None):
    print("\n--- Starting Training ---")
    
    match training_method:
        case 'svr':
            model = SVRModel()
            filename = model_filename if model_filename else 'svm.joblib'
            model_path = Path.cwd() / 'model_training' / filename

            if use_cache and model.load_model(model_path):
                print("Skipping training as cached model was loaded.")
            else:
                # For SVR, combine train and validation sets to maximize training data
                if val_dataset is not None:
                    print("Combining train and validation datasets for SVR training...")
                    combined_train_dataset = concatenate_datasets([train_dataset, val_dataset])
                else:
                    combined_train_dataset = train_dataset

                model.train(combined_train_dataset)
                model.save_model(model_path)
        
        case 'cnn':
            model = CNNModel()
            filename = model_filename if model_filename else 'cnn_age_model.keras'
            model_path = Path.cwd() / 'model_training' / filename
            
            if use_cache and model.load_model(model_path):
                print("Skipping training as cached model was loaded.")
            else:
                model.train_cnn(train_dataset, val_dataset=val_dataset, epochs=20, batch_size=64)
                model.save_model(model_path)
        
        case 'gbm':
            model = GradientBoostingModel()
            filename = model_filename if model_filename else 'gbm.joblib'
            model_path = Path.cwd() / 'model_training' / filename
            
            if use_cache and model.load_model(model_path):
                print("Skipping training as cached model was loaded.")
            else:
                model.train(train_dataset, val_dataset=val_dataset)
                model.save_model(model_path)
    
    return model

if __name__ == "__main__":
    main()
