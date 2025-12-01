from dataset_preparation.filter_italy_dataset import ItalyDataset
from dataset_preparation.filter_europe_dataset import EuropeDataset
from dataset_preparation.scene_filter import SceneFilter
from model_training.train_age_model import AgeModel
from model_training.train_cnn_model import CNNModel
from result_visualization.visualize_predictions import PredictionVisualizer
from result_visualization.visualize_feature_space import FeatureSpaceVisualizer
from result_visualization.find_worst_predictions import WorstPredictionsFinder
from result_visualization.confusion_matrix_age import AgeConfusionMatrix
from pathlib import Path

def main():
    # Initialize Dataset Handler
    italy_geojson_path = Path.cwd() / 'resources' / 'italy_borders.geojson'
    europe_geojson_path = Path.cwd() / 'resources' / 'europe_borders.geojson'
    dataset_name = "Morris0401/Year-Guessr-Dataset"
    italy_data = ItalyDataset(italy_geojson_path,dataset_name)
    
    # Load and Filter Data by Geography
    print("\n--- Loading and Filtering by Geography ---")
    train_dataset = italy_data.get_filtered_dataset(split="train")
    test_dataset = italy_data.get_filtered_dataset(split="test")
    val_dataset = italy_data.get_filtered_dataset(split="valid")
    
    # Apply Scene Parsing Filter (keep only exterior building facades)
    print("\n--- Applying Facade Detection Filter ---")
    print("Loading scene parsing model (SegFormer trained on ADE20K)...")
    scene_filter = SceneFilter()

    print("\nFiltering datasets to keep only exterior building facades:")
    print("  - Excludes interior shots (walls, floors, ceilings dominant)")
    print("  - Requires visible sky (exterior indicator)")
    
    # Visualize 5 rejected images from train dataset to show what gets filtered out
    #train_dataset = scene_filter.filter_dataset(train_dataset, verbose=True, visualize_rejected=5)
    #test_dataset = scene_filter.filter_dataset(test_dataset, verbose=True)
    #val_dataset = scene_filter.filter_dataset(val_dataset, verbose=True)

    
    if len(train_dataset) == 0:
        print("No training data found. Exiting.")
        return
        
    if len(test_dataset) == 0:
        print("No test data found. Exiting.")
        return
    
    # Choose training method: 'svr' or 'cnn'
    method = 'cnn'  # Can be "cnn" or "svr"
    if method == 'svr':
        model = AgeModel()
        model_path = Path.cwd() / 'model_training' / 'cached_model.joblib'
        
        if model.load_model(model_path):
            print("Skipping training as cached model was loaded.")
        else:
            print("\n--- Starting Training ---")
            model.train(train_dataset)
    else:
        model = CNNModel()
        model_path = Path.cwd() / 'model_training' / 'cnn_age_model.keras'
        if model.load_model(model_path):
            print("Skipping training as cached model was loaded.")
        else:
            print("\n--- Starting CNN Training ---")
            model.train_cnn(train_dataset, val_dataset=val_dataset, epochs=20, batch_size=64)
        
    # 5. Evaluate
    print("\n--- Starting Evaluation ---")
    model.evaluate(test_dataset)
    
    # 6. Save model
    model.save_model(model_path)
    


    # 7. Visualize predictions
    output_pictures = Path.cwd() / 'result_visualization' / 'pictures'
    visualizer = PredictionVisualizer(model, model_type=method)
    visualizer.visualize(test_dataset, output_pictures, num_samples=3)

    # 8. Confusion Matrix by Age Period
    print("\n--- Generating Confusion Matrix ---")
    cm_analyzer = AgeConfusionMatrix(model, model_type=method)
    cm_analyzer.compute_confusion_matrix(test_dataset, output_pictures)
    cm_analyzer.analyze_errors_by_period(test_dataset, output_pictures)

    # 9. Additional visualizations (SVR only)
    if method == 'svr':
        # Visualize Feature Space
        print("\n--- Visualizing Feature Space ---")
        feature_visualizer = FeatureSpaceVisualizer(model)
        feature_visualizer.visualize(train_dataset, output_pictures)

        # Find Worst Predictions
        print("\n--- Finding Worst Predictions ---")
        output_data = Path.cwd() / 'result_visualization' / 'data'
        worst_finder = WorstPredictionsFinder(model)
        worst_finder.find_worst(test_dataset, output_data)


if __name__ == "__main__":
    main()
