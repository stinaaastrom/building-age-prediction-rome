import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tools.generate_filename import generate_filename, get_project_root

class PeriodDistributionVisualizer:
    def __init__(self):
        self.periods = {
            0: "Romanesque & Gothic (<1400)",
            1: "Renaissance (1400-1599)",
            2: "Baroque & Rococo (1600-1779)",
            3: "Neoclassicism to Modernism (1780-1944)",
            4: "Post-War Modern & Contemporary (>=1945)"
        }

    def _get_period(self, year):
        if year is None:
            return None
        if year < 1400:
            return 0
        elif year < 1600:
            return 1
        elif year < 1780:
            return 2
        elif year < 1945:
            return 3
        else:
            return 4

    def visualize(self, dataset, save_path=None):
        """
        Visualizes the distribution of images across architectural periods.
        
        Args:
            dataset: HuggingFace dataset containing 'Year' column
            save_path: Optional path to save the plot image
        """
        print("Calculating period distribution...")
        
        # Handle both list of dicts or HF dataset
        if hasattr(dataset, 'features'):
            years = dataset['Year']
        else:
            years = [item['Year'] for item in dataset]
            
        period_counts = {k: 0 for k in self.periods.keys()}
        
        for year in years:
            period = self._get_period(year)
            if period is not None:
                period_counts[period] += 1
                
        # Prepare data for plotting
        labels = [self.periods[k] for k in sorted(self.periods.keys())]
        counts = [period_counts[k] for k in sorted(self.periods.keys())]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.bar(labels, counts, color='#4C72B0')
        
        # Customize plot
        plt.title('Distribution of Images by Architectural Period', fontsize=16, pad=20)
        plt.xlabel('Architectural Period', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.xticks(rotation=30, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
            
        plt.tight_layout()
        
        if save_path:
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        else:
            # Default save behavior
            output_path = get_project_root() / 'result_visualization' / 'pictures'
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = generate_filename('period_distribution')
            filepath = output_path / (filename + '.png')
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {filepath}")
            
        plt.close()
