import matplotlib.pyplot as plt
import random
import numpy as np
from pathlib import Path

from tools.generate_filename import generate_filename, get_project_root

class PredictionVisualizer:
    def __init__(self, model, model_type='svr'):
        """
        Initialize visualizer.
        
        Args:
            model: Either AgeModel (SVR) or CNNModel
            model_type: 'svr' or 'cnn'
        """
        self.model = model
        self.model_type = model_type

    def visualize(self, dataset, num_samples=3):

        print(f"Visualizing {num_samples} random predictions...")
        
        # Ensure we have enough samples
        if len(dataset) < num_samples:
            num_samples = len(dataset)
            
        # Pick random indices
        indices = random.sample(range(len(dataset)), num_samples)
        subset = dataset.select(indices)
        
        # Setup plot
        fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 6))
        
        if num_samples == 1:
            axes = [axes]
        
        # Process based on model type
        # Use unified predict_dataset for batch prediction if possible, 
        # but here we need per-sample visualization.
        # We can still use the model's prediction logic but applied to single items or small batch.
        
        # Let's use a small batch prediction for the subset
        if hasattr(self.model, 'predict_dataset'):
            y_pred_batch, y_true_batch, _ = self.model.predict_dataset(subset)
            
            for i, ax in enumerate(axes):
                item = subset[i]
                img = item['Picture']
                name = item.get('Building', 'No description')
                year_true = y_true_batch[i]
                year_pred = y_pred_batch[i]
                
                self._plot_prediction(ax, img, year_true, year_pred, name)
        else:
            # Fallback logic (should be removed if all models support predict_dataset)
            print("Warning: Model does not support predict_dataset. Visualization might fail.")
        
        plt.tight_layout()

        # Save figure to pictures directory
        output_path = get_project_root() / 'result_visualization' / 'pictures'
        output_path.mkdir(parents=True, exist_ok=True)
        
        filename = generate_filename('predictions')
        filepath = output_path / (filename + '.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {filepath}")
        plt.close()
    
    def _plot_prediction(self, ax, img, year_true, year_pred, name):
        """Helper method to plot a single prediction"""
        # Display Image
        ax.imshow(img)
        ax.axis('off')
        
        title_text = (
            f"True Year: {year_true}\n"
            f"Predicted: {year_pred:.0f}\n"
            f"Error: {abs(year_true - year_pred):.0f} years\n\n"
            f"Context: {name}"
        )
        
        ax.set_title(title_text, fontsize=10, wrap=True)
