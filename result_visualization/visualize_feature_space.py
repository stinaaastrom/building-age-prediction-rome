import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path
from tools.generate_filename import Filename

class FeatureSpaceVisualizer:
    def __init__(self, model):
        self.model = model

    def visualize(self, dataset, output_path: Path):
        print("Extracting features for t-SNE...")
        # We use the prepare_data method which extracts features
        X, y = self.model.prepare_data(dataset)
        
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
        
        # Ensure directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        save_path = output_path / (Filename.generate('feature_space_tsne') + '.jpg')

        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")

