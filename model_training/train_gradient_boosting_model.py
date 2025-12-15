import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from torchvision import models
from dataset_preparation.image_processing import ImageProcessing
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor


class GradientBoostingModel:
    def __init__(self):
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        self.feature_extractor = self._build_feature_extractor()
        self.scaler = StandardScaler()
        self.image_processor = ImageProcessing()
        
        print("Using sklearn GradientBoostingRegressor")
        self.model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
            verbose=1
        )

    def save_model(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        path = Path(path)
        if path.exists():
            saved_data = joblib.load(path)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            print(f"Model loaded from {path}")
            return True
        return False

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_feature_extractor(self):
        print("Loading EfficientNet V2 Large...")
        model = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.DEFAULT)
        
        # Remove classification layer
        model.classifier = nn.Identity()
        model.to(self.device)
        model.eval()
        return model

    def _extract_features_batch(self, batch, training: bool = False):
        images = []
        for img in batch['Picture']:
            try:
                images.append(self.image_processor.image_processing(img))
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        if not images:
            return {"features": []}

        input_tensor = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
        
        # Flatten
        features = features.view(features.size(0), -1).cpu().numpy()
        return {"features": features}

    def prepare_data(self, dataset, training: bool):
        print("Extracting features...")
        dataset_with_features = dataset.map(
            lambda batch: self._extract_features_batch(batch, training=training), 
            batched=True, 
            batch_size=32
        )
        
        X = np.array(dataset_with_features['features'])
        y = np.array(dataset_with_features['Year'])
        
        # Extract coordinates metadata
        lats = []
        lons = []
        for item in dataset_with_features:
            try:
                lat = float(item.get('lat_num', 0))
                lon = float(item.get('lon_num', 0))
                lats.append(lat)
                lons.append(lon)
            except (ValueError, TypeError):
                lats.append(0.0)
                lons.append(0.0)
        
        coords = np.column_stack([lats, lons])
        
        # Filter None values
        valid_mask = [y_val is not None for y_val in y]
        X = X[valid_mask]
        y = y[valid_mask]
        coords = coords[valid_mask]
        
        # Concatenate image features with coordinate metadata
        X_combined = np.concatenate([X, coords], axis=1)
        print(f"Feature dimensions: {X.shape[1]} image + 2 coords = {X_combined.shape[1]} total")
        
        return X_combined, y

    def train(self, train_dataset, val_dataset=None):
        
        # Combine train and validation sets to maximize training data if using sklearn
        if val_dataset is not None:
            print("Combining train and validation datasets for GBM training...")
            from datasets import concatenate_datasets
            train_dataset = concatenate_datasets([train_dataset, val_dataset])
        
        print("Preparing training data...")
        X_train, y_train = self.prepare_data(train_dataset, training=True)
        
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print(f"Training Gradient Boosting on {len(X_train)} samples...")
        self.model.fit(X_train_scaled, y_train)
        
        print("Training complete.")
        
        # Print feature importance summary (top 10 features)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-10:][::-1]
            print("\nTop 10 Feature Importances:")
            for idx in top_indices:
                print(f"  Feature {idx}: {importances[idx]:.4f}")

    def evaluate(self, test_dataset):
        print("Preparing test data...")
        X_test, y_test = self.prepare_data(test_dataset, training=False)
        
        print(f"Evaluating on {len(X_test)} samples...")
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae:.2f} years")
        
        # Additional metrics
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f} years")
        
        # Percentage within X years
        within_10 = np.mean(np.abs(y_test - y_pred) <= 10) * 100
        within_20 = np.mean(np.abs(y_test - y_pred) <= 20) * 100
        within_50 = np.mean(np.abs(y_test - y_pred) <= 50) * 100
        print(f"Within 10 years: {within_10:.1f}%")
        print(f"Within 20 years: {within_20:.1f}%")
        print(f"Within 50 years: {within_50:.1f}%")
        
        print("\nExample Predictions:")
        for i in range(min(5, len(y_test))):
            print(f"True: {y_test[i]}, Predicted: {y_pred[i]:.1f}, Error: {abs(y_test[i] - y_pred[i]):.1f}")
        
        return mae

    def predict_dataset(self, dataset):
        """
        Predicts years for a given dataset.
        Returns:
            y_pred: Predicted years
            y_true: Actual years
            coords: Coordinates (lat, lon)
        """
        X, y_true = self.prepare_data(dataset, training=False)
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        # Extract coords from X (last 2 columns)
        coords = X[:, -2:] 
        
        return y_pred, y_true, coords
