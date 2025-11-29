import matplotlib.pyplot as plt
import random
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime

from tools.generate_filename import Filename

class PredictionVisualizer:
    def __init__(self, model):
        self.model = model

    def visualize(self, dataset, output_path: Path, num_samples=3):

        

        print(f"Visualizing {num_samples} random predictions...")
        
        # Ensure we have enough samples
        if len(dataset) < num_samples:
            num_samples = len(dataset)
            
        # Pick random indices
        indices = random.sample(range(len(dataset)), num_samples)
        subset = dataset.select(indices)
        
        # Extract features for this subset
        # We use the model's internal method to get features
        subset_with_features = subset.map(
            self.model._extract_features_batch, 
            batched=True, 
            batch_size=num_samples
        )
        
        # Setup plot
        fig, axes = plt.subplots(1, num_samples, figsize=(5 * num_samples, 6))
        
        if num_samples == 1:
            axes = [axes]
            
        for i, ax in enumerate(axes):
            item = subset_with_features[i]
            features = item['features']
            year_true = item['Year']
            img = item['Picture']
            name = item.get('Building', 'No description')
            
            # Handle missing data
            if year_true is None:
                ax.text(0.5, 0.5, "No Year Data", ha='center')
                ax.axis('off')
                continue
                
            if not features:
                ax.text(0.5, 0.5, "Feature Extraction Failed", ha='center')
                ax.axis('off')
                continue

            # Predict
            # SVR expects 2D array (1, n_features)
            feat_vector = np.array(features).reshape(1, -1)
            year_pred = self.model.svr.predict(feat_vector)[0]
            
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
            
        plt.tight_layout()

        # Ensure output_path is a directory. If it exists but is a file -> error.
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f"Output path '{output_path}' exists and is not a directory")

        # Create directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamped filename with git hash
        filename = Filename.generate('predictions') + ".jpg"
        save_path = output_path / filename

        # Save figure
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
