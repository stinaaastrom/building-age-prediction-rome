from filter_rome_dataset import RomeDataset
from train_age_model import AgeModel
from visualize_predictions import PredictionVisualizer

def main():
    # 1. Initialize Dataset Handler
    rome_data = RomeDataset()
    
    # 2. Load and Filter Data
    train_dataset = rome_data.get_filtered_dataset(split="train")
    val_dataset = rome_data.get_filtered_dataset(split="valid")
    test_dataset = rome_data.get_filtered_dataset(split="test")
    
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
        print("\n--- Starting SVR Training ---")
        model.train(train_dataset)
        
        # 5. Evaluate
        print("\n--- Starting SVR Evaluation ---")
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
