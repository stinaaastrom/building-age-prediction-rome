"""
Confusion matrix visualization for building age prediction.
Groups predictions into age periods to analyze model performance across different eras.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from tools.generate_filename import generate_filename, get_project_root


class AgeConfusionMatrix:
    """
    Compute and visualize confusion matrix for age prediction grouped by periods.
    """
    
    def __init__(self, model, model_type='svr'):
        """
        Initialize confusion matrix generator.
        
        Args:
            model: Trained model (AgeModel for SVR or CNNModel for CNN)
            model_type: 'svr' or 'cnn'
        """
        self.model = model
        self.model_type = model_type
        
        # Define age periods for grouping
        # Use None for open-ended ranges (no beginning/end)
        self.age_periods = [
            (None, 1400, "Romanesque & Gothic (<1400)"),
            (1400, 1600, "Renaissance (1400-1599)"),
            (1600, 1780, "Baroque & Rococo (1600-1779)"),
            (1780, 1945, "Neoclassicism to Modernism (1780-1944)"),
            (1945, None, "Post-War Modern & Contemporary (>=1945)")
        ]
    
    def _assign_period(self, year):
        """
        Assign a year to an age period.
        
        Args:
            year: Year to classify
            
        Returns:
            Period index (0-5) or -1 if out of range
        """
        for idx, (start, end, _) in enumerate(self.age_periods):
            # Handle open-ended ranges (None means no limit)
            start_ok = start is None or year >= start
            end_ok = end is None or year < end
            
            if start_ok and end_ok:
                return idx
        return -1  # Out of range
    
    def _get_predictions_svr(self, dataset):
        """Get predictions using SVR model."""
        X_test, y_test = self.model.prepare_data(dataset, training=False)
        
        # Scale features if scaler exists
        if hasattr(self.model, 'scaler'):
            X_test = self.model.scaler.transform(X_test)
            
        y_pred = self.model.svr.predict(X_test)
        return y_test, y_pred
    
    def _get_predictions_cnn(self, dataset):
        """Get predictions using CNN model."""
        X_test_imgs, X_test_coords, y_test_norm = self.model.prepare_data(
            dataset, 
            image_size=(224, 224), 
            use_augmentation=False
        )
        
        predictions_norm = self.model.cnn_model.predict([X_test_imgs, X_test_coords], verbose=0).flatten()
        predictions = self.model.denormalize_year(predictions_norm)
        y_test = self.model.denormalize_year(y_test_norm)
        
        return y_test, predictions
    
    def compute_confusion_matrix(self, dataset):
        """
        Compute and visualize confusion matrix for age periods.
        
        Args:
            dataset: Test dataset
        """
        print("\n--- Computing Confusion Matrix for Age Periods ---")
        
        # Get predictions based on model type
        if self.model_type == 'svr':
            y_true, y_pred = self._get_predictions_svr(dataset)
            # SVR predicts years, so we map them to periods
            y_true_periods = np.array([self._assign_period(year) for year in y_true])
            y_pred_periods = np.array([self._assign_period(year) for year in y_pred])
        else:
            y_true, y_pred = self._get_predictions_cnn(dataset)
            # CNN still predicts years, so we map them
            y_true_periods = np.array([self._assign_period(year) for year in y_true])
            y_pred_periods = np.array([self._assign_period(year) for year in y_pred])
        
        # Filter out any out-of-range predictions
        valid_mask = (y_true_periods >= 0) & (y_pred_periods >= 0)
        y_true_periods = y_true_periods[valid_mask]
        y_pred_periods = y_pred_periods[valid_mask]
        
        # Get period labels
        period_labels = [label for _, _, label in self.age_periods]
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_periods, y_pred_periods)
        
        # Compute normalized confusion matrix (by row)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        
        # Plot absolute counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=period_labels, yticklabels=period_labels,
                   ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix - Absolute Counts', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Period', fontsize=12)
        axes[0].set_xlabel('Predicted Period', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].tick_params(axis='y', rotation=0)
        
        # Plot normalized percentages
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=period_labels, yticklabels=period_labels,
                   ax=axes[1], cbar_kws={'label': 'Percentage'})
        axes[1].set_title('Confusion Matrix - Normalized (Row %)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Period', fontsize=12)
        axes[1].set_xlabel('Predicted Period', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        # Save figure to pictures directory
        output_path = get_project_root() / 'result_visualization' / 'pictures'
        output_path.mkdir(parents=True, exist_ok=True)
        filename = generate_filename('confusion_matrix')
        filepath = output_path / (filename + '.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {filepath}")
        plt.show()
        plt.close()
        
        # Print classification report
        print("\n--- Classification Report by Age Period ---")
        print(classification_report(y_true_periods, y_pred_periods, 
                                   target_names=period_labels, 
                                   zero_division=0))
        
        # Print per-period statistics
        print("\n--- Per-Period Statistics ---")
        for idx, (start, end, label) in enumerate(self.age_periods):
            mask_true = y_true_periods == idx
            mask_pred = y_pred_periods == idx
            
            n_true = mask_true.sum()
            n_pred = mask_pred.sum()
            n_correct = ((y_true_periods == idx) & (y_pred_periods == idx)).sum()
            
            if n_true > 0:
                accuracy = n_correct / n_true * 100
                print(f"\n{label}:")
                print(f"  True samples: {n_true}")
                print(f"  Predicted as this period: {n_pred}")
                print(f"  Correctly classified: {n_correct} ({accuracy:.1f}%)")
                
                # Calculate average error for this period
                period_mask = (y_true_periods == idx)
                if period_mask.sum() > 0:
                    y_true_filtered = y_true[valid_mask][period_mask]
                    y_pred_filtered = y_pred[valid_mask][period_mask]
                    mae = np.mean(np.abs(y_true_filtered - y_pred_filtered))
                    print(f"  Mean Absolute Error: {mae:.1f} years")
        
        return cm, cm_normalized
    
    def analyze_errors_by_period(self, dataset, top_n=10):
        """
        Analyze which periods have the largest errors.
        
        Args:
            dataset: Test dataset
            output_path: Directory to save visualization
            top_n: Number of worst predictions to show per period
        """
        print("\n--- Analyzing Errors by Period ---")
        
        # Get predictions
        if self.model_type == 'svr':
            y_true, y_pred = self._get_predictions_svr(dataset)
        else:
            y_true, y_pred = self._get_predictions_cnn(dataset)
        
        # Calculate errors
        errors = np.abs(y_true - y_pred)
        y_true_periods = np.array([self._assign_period(year) for year in y_true])
        
        # Create box plot of errors by period
        fig, ax = plt.subplots(figsize=(12, 6))
        
        period_errors = []
        period_labels = []
        for idx, (start, end, label) in enumerate(self.age_periods):
            mask = y_true_periods == idx
            if mask.sum() > 0:
                period_errors.append(errors[mask])
                period_labels.append(label)
        
        bp = ax.boxplot(period_errors, labels=period_labels, patch_artist=True)
        
        # Color the boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        ax.set_ylabel('Absolute Error (years)', fontsize=12)
        ax.set_xlabel('Age Period', fontsize=12)
        ax.set_title('Distribution of Prediction Errors by Age Period', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure to pictures directory
        output_path = get_project_root() / 'result_visualization' / 'pictures'
        output_path.mkdir(parents=True, exist_ok=True)
        filename = generate_filename('error_by_period')
        filepath = output_path / (filename + '.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved error distribution to {filepath}")
        plt.show()
        plt.close()
