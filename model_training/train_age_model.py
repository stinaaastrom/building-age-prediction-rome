import torch
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
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

class AgeModel:
    def __init__(self):
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        self.feature_extractor = self._build_feature_extractor()
        self.svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        # Separate processors: training uses augmentation, eval/test is deterministic
        self.image_processor = ImageProcessing()

    def save_model(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.svr, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        path = Path(path)
        if path.exists():
            self.svr = joblib.load(path)
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
        print("Loading ResNet50...")
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        
        # Use standard 3-channel RGB input (no modification needed)
        
        # Remove classification layer
        modules = list(model.children())[:-1]
        extractor = nn.Sequential(*modules)
        extractor.to(self.device)
        extractor.eval()
        return extractor

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
        
        # Filter None values
        valid_mask = [y_val is not None for y_val in y]
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Feature dimensions: {X.shape[1]} image features")
        
        return X, y

    def train(self, train_dataset):
        print("Preparing training data...")
        X_train, y_train = self.prepare_data(train_dataset, training=True)
        
        print(f"Training SVR on {len(X_train)} samples...")
        self.svr.fit(X_train, y_train)
        print("Training complete.")

    def evaluate(self, test_dataset):
        print("Preparing test data...")
        X_test, y_test = self.prepare_data(test_dataset, training=False)
        
        print(f"Evaluating on {len(X_test)} samples...")
        y_pred = self.svr.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae:.2f} years")
        
        print("\nExample Predictions:")
        for i in range(min(5, len(y_test))):
            print(f"True: {y_test[i]}, Predicted: {y_pred[i]:.1f}, Error: {abs(y_test[i] - y_pred[i]):.1f}")
        
        return mae