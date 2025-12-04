import torch
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from torchvision import models
from dataset_preparation.image_processing import ImageProcessing
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from pathlib import Path

class SVRModel:
    def __init__(self):
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        self.feature_extractor = self._build_feature_extractor()
        self.svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        self.scaler = StandardScaler()
        # Separate processors: training uses augmentation, eval/test is deterministic
        self.image_processor = ImageProcessing()

    def save_model(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'svr': self.svr, 'scaler': self.scaler}, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        path = Path(path)
        if path.exists():
            saved_data = joblib.load(path)
            self.svr = saved_data['svr']
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
        
        # Use standard 3-channel RGB input (no modification needed)
        
        # Remove classification layer (classifier is the last module in ConvNeXt)
        # ConvNeXt structure: features -> avgpool -> classifier
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
                # Add a dummy tensor or handle gracefully? 
                # For simplicity, we might skip, but batch processing expects same size.
                # We'll just append a zero tensor of correct shape if needed, 
                # but here we assume most images are fine.
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

    def train(self, train_dataset):
        print("Preparing training data...")
        X_train, y_train = self.prepare_data(train_dataset, training=True)
        
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print(f"Training SVR on {len(X_train)} samples...")
        self.svr.fit(X_train_scaled, y_train)
        print("Training complete.")

    def evaluate(self, test_dataset):
        print("Preparing test data...")
        X_test, y_test = self.prepare_data(test_dataset, training=False)
        
        print(f"Evaluating on {len(X_test)} samples...")
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.svr.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae:.2f} years")
        
        print("\nExample Predictions:")
        for i in range(min(5, len(y_test))):
            print(f"True: {y_test[i]}, Predicted: {y_pred[i]:.1f}, Error: {abs(y_test[i] - y_pred[i]):.1f}")
        
        return mae