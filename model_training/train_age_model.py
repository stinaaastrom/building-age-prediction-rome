import torch
import torch.nn as nn
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
import numpy as np
from torchvision import models
from image_processing import ImageProcessing
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
        self.train_image_processor = ImageProcessing(use_augmentation=True)
        self.eval_image_processor = ImageProcessing(use_augmentation=False)

        # CNN model attributes
        self.cnn_model = None
        self.history = None
        self.year_min = 1800.0
        self.year_max = 2020.0

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

    # CNN Methods
    def _normalize_year(self, year):
        """Normalize year to [0, 1] range"""
        return (year - self.year_min) / (self.year_max - self.year_min)
    
    def _denormalize_year(self, normalized_year):
        """Denormalize year back to original range"""
        return normalized_year * (self.year_max - self.year_min) + self.year_min

    def _prepare_cnn_data(self, dataset, image_size=(128, 128)):
        """Prepare images and labels for CNN"""
        print(f"Preparing CNN data with image size {image_size}...")
        X = []
        y = []
        
        for item in dataset:
            try:
                img = item['Picture']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img = img.resize(image_size)
                img_array = np.array(img) / 255.0
                
                X.append(img_array)
                y.append(self._normalize_year(item['Year']))
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        valid_mask = [y_val is not None for y_val in y]
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"Prepared {len(X)} samples with shape {X.shape}")
        return X, y

    def _build_cnn_model(self, input_shape=(128, 128, 3)):
        """Build simplified CNN model for faster training"""
        print("Building CNN model...")
        input_layer = Input(shape=input_shape)
        
        # Block 1
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Block 2
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.25)(x)
        
        # Block 3
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(0.3)(x)
        
        # FC layers
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)

        age_output = Dense(1, activation='sigmoid', name='age_output')(x)

        self.cnn_model = Model(inputs=input_layer, outputs=age_output)
        self.cnn_model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.001),
            metrics=['mae']
        )
        self.cnn_model.summary()

    def train_cnn(self, train_dataset, val_dataset=None, epochs=20, batch_size=64, image_size=(128, 128)):
        """Train using CNN model (optimized for speed)"""
        # Prepare data
        X_train, y_train = self._prepare_cnn_data(train_dataset, image_size=image_size)
        
        if val_dataset is not None:
            X_val, y_val = self._prepare_cnn_data(val_dataset, image_size=image_size)
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Build model
        self._build_cnn_model(input_shape=(image_size[0], image_size[1], 3))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
        ]
        
        # Train
        print(f"\nTraining CNN for {epochs} epochs...")
        self.history = self.cnn_model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        print("Training complete.")

    def evaluate_cnn(self, test_dataset, image_size=(128, 128)):
        """Evaluate CNN model"""
        if self.cnn_model is None:
            print("No CNN model trained!")
            return None
        
        print("Evaluating CNN model...")
        X_test, y_test_norm = self._prepare_cnn_data(test_dataset, image_size=image_size)
        
        loss, mae_norm = self.cnn_model.evaluate(X_test, y_test_norm, verbose=0)
        
        # Denormalize for actual MAE
        predictions_norm = self.cnn_model.predict(X_test, verbose=0).flatten()
        predictions = self._denormalize_year(predictions_norm)
        y_test = self._denormalize_year(y_test_norm)
        
        mae_actual = np.mean(np.abs(y_test - predictions))
        
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test MAE: {mae_actual:.2f} years")
        
        print("\nExample Predictions:")
        for i in range(min(5, len(y_test))):
            print(f"True: {y_test[i]:.0f}, Predicted: {predictions[i]:.1f}, Error: {abs(y_test[i] - predictions[i]):.1f}")
        
        return mae_actual

    def save_cnn_model(self, path='cnn_age_model.keras'):
        """Save CNN model"""
        if self.cnn_model is not None:
            self.cnn_model.save(path)
            print(f"CNN model saved to {path}")

    def load_cnn_model(self, path='cnn_age_model.keras'):
        """Load CNN model"""
        self.cnn_model = keras.models.load_model(path)
        print(f"CNN model loaded from {path}")
