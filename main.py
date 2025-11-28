from dataset_preparation.filter_italy_dataset import ItalyDataset
from dataset_preparation.filter_europe_dataset import EuropeDataset
from model_training.train_age_model import AgeModel
from result_visualization.visualize_predictions import PredictionVisualizer
from result_visualization.visualize_feature_space import FeatureSpaceVisualizer
from result_visualization.find_worst_predictions import WorstPredictionsFinder
from pathlib import Path

def main():
    # Initialize Dataset Handler
    italy_geojson_path = Path.cwd() / 'resources' / 'italy_borders.geojson'
    europe_geojson_path = Path.cwd() / 'resources' / 'europe_borders.geojson'
    dataset_name = "Morris0401/Year-Guessr-Dataset"
    italy_data = ItalyDataset(italy_geojson_path,dataset_name)
    
    # Load and Filter Data
    train_dataset = italy_data.get_filtered_dataset(split="train")
    test_dataset = italy_data.get_filtered_dataset(split="test")
    val_dataset = rome_data.get_filtered_dataset(split="valid")

    
    if len(train_dataset) == 0:
        print("No training data found. Exiting.")
        return
        
    if len(test_dataset) == 0:
        print("No test data found. Exiting.")
        return

    # 3. Initialize Model
    model = AgeModel()
    
    # Choose training method: 'svr' or 'cnn'
    method = 'cnn'  # Change to 'cnn' to train with CNN
    
    if method == 'svr':
        # 4. Train SVR
        # 4. Train or Load
        model_path = Path.cwd() / 'model_training' / 'cached_model.joblib'
        
        if model.load_model(model_path):
            print("Skipping training as cached model was loaded.")
        else:
            print("\n--- Starting Training ---")
            model.train(train_dataset)
            model.save_model(model_path)
    
        # 5. Evaluate
        print("\n--- Starting Evaluation ---")
        model.evaluate(test_dataset)
    
    elif method == 'cnn':
        # 4. Train CNN (faster training with smaller images and fewer epochs)
        print("\n--- Starting CNN Training ---")
        model.train_cnn(train_dataset, val_dataset=val_dataset, epochs=20, batch_size=64, image_size=(128, 128))
        
        # 5. Evaluate
        print("\n--- Starting CNN Evaluation ---")
        model.evaluate_cnn(test_dataset)
        
        # 6. Save model
        model.save_cnn_model('cnn_age_model.keras')

    # 7. Visualize (only works with SVR currently)
    if method == 'svr':
        print("\n--- Visualizing Predictions ---")
        visualizer = PredictionVisualizer(model)
        visualizer.visualize(test_dataset, num_samples=3)

if __name__ == "__main__":
    main()
