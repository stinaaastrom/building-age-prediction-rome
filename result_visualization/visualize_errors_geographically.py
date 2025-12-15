"""
Geographic visualization of prediction errors across Italy.
Displays each building as a point on a map, colored by prediction error magnitude.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
from tools.generate_filename import generate_filename, get_project_root


class GeographicErrorVisualizer:
    """Visualize prediction errors geographically on a map of Italy."""
    
    def __init__(self, model):
        """
        Initialize the geographic error visualizer.
        
        Args:
            model: The trained model (SVR or CNN)
        """
        self.model = model
    
    def visualize_errors_on_map(self, dataset):
        """
        Create a scatter plot map showing prediction errors for each building location.
        Lighter colors = smaller errors, darker colors = larger errors.
        
        Args:
            dataset: The dataset with images, coordinates, and years
        """
        print("Extracting predictions and coordinates...")
        
        # Get features and labels
        # Get predictions and coordinates
        y_pred, y_true, coords = self.model.predict_dataset(dataset)
        lats = coords[:, 0]
        lons = coords[:, 1]

        # Calculate absolute errors
        errors = np.abs(y_true - y_pred)
        
        print(f"Extracted {len(errors)} predictions")
        print(f"Error statistics - Min: {errors.min():.2f}, Max: {errors.max():.2f}, Mean: {errors.mean():.2f}")
        lons = np.array(lons)
        
        # Filter valid coordinates (those in Italy region approximately)
        valid_mask = (lats > 36) & (lats < 48) & (lons > 6) & (lons < 19)
        
        lats_valid = lats[valid_mask]
        lons_valid = lons[valid_mask]
        errors_valid = errors[valid_mask]
        
        print(f"Valid locations in Italy: {len(lats_valid)}/{len(lats)}")
        
        # Check if we have any valid locations
        if len(lats_valid) == 0:
            print("ERROR: No valid coordinates found in dataset. Skipping geographic visualization.")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Use fixed color thresholds: 0-100 (green), 100-200 (yellow), 200+ (red)
        # Normalize to 0-300 range to show the three categories clearly
        norm = Normalize(vmin=0, vmax=300)
        cmap = plt.cm.RdYlGn_r  # Red for large errors, Green for small errors
        
        # Create scatter plot
        scatter = ax.scatter(
            lons_valid, lats_valid,
            c=errors_valid,
            cmap=cmap,
            norm=norm,
            s=100,
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Prediction Error (years)', pad=0.02)
        
        # Set labels and title
        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        ax.set_title('Geographic Distribution of Building Age Prediction Errors in Italy', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set Italy boundaries (approximate)
        ax.set_xlim(6, 19)
        ax.set_ylim(36, 48)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add Italy borders
        try:
            import json
            from pathlib import Path
            from shapely.geometry import shape
            
            italy_geojson_path = Path.cwd() / 'resources' / 'italy_borders.geojson'
            if italy_geojson_path.exists():
                with open(italy_geojson_path, 'r') as f:
                    data = json.load(f)
                italy_feature = data['features'][0]
                italy_polygon = shape(italy_feature['geometry'])
                
                # Plot the border
                if italy_polygon.geom_type == 'Polygon':
                    x, y = italy_polygon.exterior.xy
                    ax.plot(x, y, color='black', linewidth=2, zorder=1)
                elif italy_polygon.geom_type == 'MultiPolygon':
                    for poly in italy_polygon.geoms:
                        x, y = poly.exterior.xy
                        ax.plot(x, y, color='black', linewidth=2, zorder=1)
        except Exception as e:
            print(f"Note: Could not load Italy borders: {e}")
        
        # Add legend explaining colors with specific thresholds
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', edgecolor='black', label='Small error (0-100 years)'),
            Patch(facecolor='yellow', edgecolor='black', label='Medium error (100-200 years)'),
            Patch(facecolor='red', edgecolor='black', label='Large error (200+ years)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Save figure
        output_path = get_project_root() / 'result_visualization' / 'pictures'
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / (generate_filename('geographic_errors_italy') + '.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Geographic error map saved to {save_path}")
        
        return errors_valid
    
    def visualize_error_density_regions(self, dataset):
        """
        Create a regional view showing error statistics by geographic regions.
        
        Args:
            dataset: The dataset with images, coordinates, and years
        """
        print("Creating regional error density visualization...")
        
        # Get predictions and coordinates
        y_pred, y_true, coords = self.model.predict_dataset(dataset)
        lats = coords[:, 0]
        lons = coords[:, 1]
        
        # Calculate absolute errors
        errors = np.abs(y_true - y_pred)
        
        # Filter valid coordinates
        valid_mask = (lats > 36) & (lats < 48) & (lons > 6) & (lons < 19)
        
        lats_valid = lats[valid_mask]
        lons_valid = lons[valid_mask]
        errors_valid = errors[valid_mask]
        
        # Check if we have any valid locations
        if len(lats_valid) == 0:
            print("ERROR: No valid coordinates found in dataset. Skipping regional density visualization.")
            return None
        
        # Create 2D histogram/heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create hexbin plot (more suitable for geographic data)
        hexbin = ax.hexbin(
            lons_valid, lats_valid,
            C=errors_valid,
            gridsize=15,
            cmap='RdYlGn_r',
            mincnt=1,
            reduce_C_function=np.mean,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(hexbin, ax=ax, label='Mean Prediction Error (years)')
        
        # Set labels and title
        ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
        ax.set_title('Mean Prediction Error by Region in Italy', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set Italy boundaries
        ax.set_xlim(6, 19)
        ax.set_ylim(36, 48)
        
        # Add Italy borders
        try:
            import json
            from pathlib import Path
            from shapely.geometry import shape
            
            italy_geojson_path = Path.cwd() / 'resources' / 'italy_borders.geojson'
            if italy_geojson_path.exists():
                with open(italy_geojson_path, 'r') as f:
                    data = json.load(f)
                italy_feature = data['features'][0]
                italy_polygon = shape(italy_feature['geometry'])
                
                # Plot the border
                if italy_polygon.geom_type == 'Polygon':
                    x, y = italy_polygon.exterior.xy
                    ax.plot(x, y, color='black', linewidth=2, zorder=1)
                elif italy_polygon.geom_type == 'MultiPolygon':
                    for poly in italy_polygon.geoms:
                        x, y = poly.exterior.xy
                        ax.plot(x, y, color='black', linewidth=2, zorder=1)
        except Exception as e:
            print(f"Note: Could not load Italy borders: {e}")
        
        # Save figure
        output_path = get_project_root() / 'result_visualization' / 'pictures'
        output_path.mkdir(parents=True, exist_ok=True)
        save_path = output_path / (generate_filename('error_density_regions_italy') + '.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Regional error density map saved to {save_path}")
