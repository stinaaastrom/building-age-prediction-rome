import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from tools.generate_filename import generate_filename, get_project_root
from tensorflow import keras

class FeatureSpaceVisualizer:
    def __init__(self, model):
        self.model = model

    def visualize(self, dataset):
        print("Extracting features for t-SNE...")
        
        # Extract features based on model type
        if hasattr(self.model, 'prepare_data'):
            # SVR and GBM have prepare_data which returns features + coords
            X, y = self.model.prepare_data(dataset, training=False)
        elif hasattr(self.model, 'prepare_cnn_data'):
            # CNN has prepare_cnn_data which returns images, coords, labels
            # We need to extract features from the CNN backbone
            print("Extracting features from CNN backbone...")
            X_images, X_coords, y_norm = self.model.prepare_cnn_data(dataset, fit_stats=False)
            
            # Create a feature extractor model from the CNN
            # Assuming self.model.cnn_model is the Keras model
            # We want the output of the dense layer before the final prediction
            # Or the output of the DenseNet backbone.
            # Let's try to get the output of the 'concatenate' layer or the global average pooling
            
            try:
                # Find the layer before the final dense layers
                # This depends on the exact architecture in train_cnn_model.py
                # Let's assume we can get features from the layer named 'concatenate' or similar
                # Or just use the backbone output if accessible.
                
                # Safer approach: Create a new model that outputs the features
                # The model structure in train_cnn_model.py is:
                # inputs -> backbone -> pooling -> dense -> concat -> dense -> output
                
                # Let's try to get the output of the concatenation of image features and coords
                # If that's hard, we can just use the image features from the backbone.
                
                # For visualization purposes, let's use the penultimate layer output
                layer_name = self.model.cnn_model.layers[-2].name
                feature_extractor = keras.models.Model(
                    inputs=self.model.cnn_model.inputs,
                    outputs=self.model.cnn_model.get_layer(layer_name).output
                )
                X = feature_extractor.predict([X_images, X_coords], verbose=0)
                y = self.model.denormalize_year(y_norm)
                
            except Exception as e:
                print(f"Could not extract CNN features: {e}")
                return
        else:
            print("Model does not support feature extraction.")
            return
        
        print(f"Extracted features shape: {X.shape}")
        
        # Dimensionality Reduction
        print("Running PCA...")
        pca = PCA(n_components=50) # Reduce to 50 first for t-SNE stability
        X_pca = pca.fit_transform(X)
        
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_pca)
        
        # Visualization
        print("Plotting...")
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Year')
        plt.title(f't-SNE Visualization of ResNet50 Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        # Save figure to pictures directory
        output_path = get_project_root() / 'result_visualization' / 'pictures'
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / (generate_filename('feature_space_tsne') + '.jpg')

        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")

