import torch
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import numpy as np
from torchvision import models
from image_processing import ImageProcessing
import joblib
from pathlib import Path

class AgeModel:
    def __init__(self):
        self.device = self._get_device()
        print(f"Using device: {self.device}")
        self.feature_extractor = self._build_feature_extractor()
        self.svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
        # Separate processors: training uses augmentation, eval/test is deterministic
        self.train_image_processor = ImageProcessing(use_augmentation=True)
        self.eval_image_processor = ImageProcessing(use_augmentation=False)

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
        
        # Modify first conv layer to accept 5 channels (RGB + Sobel + Laplacian)
        original_conv1 = model.conv1
        model.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Initialize new conv layer weights
        # Copy RGB weights; channels 4 and 5 get mean of original weights
        with torch.no_grad():
            model.conv1.weight[:, :3, :, :] = original_conv1.weight
            avg_channel = original_conv1.weight.mean(dim=1, keepdim=True)
            model.conv1.weight[:, 3:4, :, :] = avg_channel
            model.conv1.weight[:, 4:5, :, :] = avg_channel
        
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
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                processor = self.train_image_processor if training else self.eval_image_processor
                images.append(processor.image_processing(img))
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
