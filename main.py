from dataset_preparation.filter_italy_dataset import ItalyDataset
from model_training.train_age_model import AgeModel
from result_visualization.visualize_predictions import PredictionVisualizer
from pathlib import Path

def main():
    # Initialize Dataset Handler
    geojson_path = Path.cwd() / 'resources' / 'italy_borders.geojson'
    dataset_name = "Morris0401/Year-Guessr-Dataset"
    italy_data = ItalyDataset(geojson_path,dataset_name)
    
    # Load and Filter Data
    train_dataset = italy_data.get_filtered_dataset(split="train")
    test_dataset = italy_data.get_filtered_dataset(split="test")
    
    if len(train_dataset) == 0:
        print("No training data found. Exiting.")
        return
        
    if len(test_dataset) == 0:
        print("No test data found. Exiting.")
        return

    # 3. Initialize Model
    model = AgeModel()
    
    # 4. Train
    print("\n--- Starting Training ---")
    model.train(train_dataset)
    
    # 5. Evaluate
    print("\n--- Starting Evaluation ---")
    model.evaluate(test_dataset)

    # 6. Visualize
    print("\n--- Visualizing Predictions ---")
    output = Path.cwd() / 'result_visualization' / 'pictures'
    visualizer = PredictionVisualizer(model)
    visualizer.visualize(test_dataset, output, num_samples=3)

if __name__ == "__main__":
    main()
