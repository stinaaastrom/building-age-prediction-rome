import matplotlib.pyplot as plt
import random
import numpy as np
import subprocess
from pathlib import Path
from datetime import datetime

from tools.generate_filename import Filename

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

    def visualize(self, dataset, output_path: Path, num_samples=3):

        

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
        if self.model_type == 'svr':
            # Extract features for SVR
            subset_with_features = subset.map(
                self.model._extract_features_batch, 
                batched=True, 
                batch_size=num_samples
            )
            
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

                # Add coordinate features
                lat = float(item.get('lat_num', 0))
                lon = float(item.get('lon_num', 0))
                coords = np.array([lat, lon])
                
                # Combine image features with coordinates
                feat_vector = np.concatenate([features, coords])
                
                # Predict
                # SVR expects 2D array (1, n_features)
                feat_vector = feat_vector.reshape(1, -1)
                year_pred = self.model.svr.predict(feat_vector)[0]
                
                self._plot_prediction(ax, img, year_true, year_pred, name)
                
        else:  # CNN
            for i, ax in enumerate(axes):
                item = subset[i]
                year_true = item['Year']
                img = item['Picture']
                name = item.get('Building', 'No description')
                
                # Handle missing data
                if year_true is None:
                    ax.text(0.5, 0.5, "No Year Data", ha='center')
                    ax.axis('off')
                    continue
                
                try:
                    # Prepare single image and coordinates for CNN (channel-last format)
                    img_array = self.model.image_processor.image_processing_for_cnn(img)
                    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
                    
                    lat = float(item.get('lat_num', 0))
                    lon = float(item.get('lon_num', 0))
                    coords = np.array([[lat, lon]])
                    
                    # Predict with CNN (returns normalized year)
                    year_pred_norm = self.model.cnn_model.predict([img_array, coords], verbose=0)[0][0]
                    year_pred = self.model._denormalize_year(year_pred_norm)
                    
                    self._plot_prediction(ax, img, year_true, year_pred, name)
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f"Prediction Failed\n{str(e)}", ha='center')
                    ax.axis('off')
                    continue
        
        plt.tight_layout()

        # Ensure output_path is a directory
        if output_path.exists() and not output_path.is_dir():
            raise ValueError(f"Output path '{output_path}' exists and is not a directory")

        # Create directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        filename = Filename.generate('predictions')
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
