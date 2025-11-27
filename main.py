from filter_rome_dataset import RomeDataset
from train_age_model import AgeModel
from visualize_predictions import PredictionVisualizer

def main():
    # 1. Initialize Dataset Handler
    rome_data = RomeDataset()
    
    # 2. Load and Filter Data
    # Using split="train" for training data
    train_dataset = rome_data.get_filtered_dataset(split="train")
    
    # Using split="test" for testing data (as requested)
    test_dataset = rome_data.get_filtered_dataset(split="test")
    
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
    visualizer = PredictionVisualizer(model)
    visualizer.visualize(test_dataset, num_samples=3)

if __name__ == "__main__":
    main()
