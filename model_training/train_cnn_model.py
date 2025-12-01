import numpy as np
from dataset_preparation.image_processing import ImageProcessing
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications import DenseNet121
from sklearn.metrics import mean_absolute_error
from keras.layers import Concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CNNModel:
    def __init__(self):
        # CNN model attributes
        self.image_processor = ImageProcessing()
        self.cnn_model = None
        self.history = None
        self.year_min = 1800.0
        self.year_max = 2020.0

    def _normalize_year(self, year):
        """Normalize year to [0, 1] range"""
        return (year - self.year_min) / (self.year_max - self.year_min)
    
    def _denormalize_year(self, normalized_year):
        """Denormalize year back to original range"""
        return normalized_year * (self.year_max - self.year_min) + self.year_min

    def _prepare_cnn_data(self, dataset, image_size=(128, 128), use_augmentation=False):
        """Prepare images, coordinates, and labels for CNN"""
        aug_text = " with augmentation" if use_augmentation else ""
        print(f"Preparing CNN data{aug_text} with image size {image_size}...")
        X_images = []
        X_coords = []
        y = []
        
        for item in dataset:
            try:
                img = item['Picture']
                # Use CNN-specific processing (channel-last format)
                img_array = self.image_processor.image_processing_for_cnn(img, use_augmentation=use_augmentation)
                
                # Extract coordinates
                lat = float(item.get('lat_num', 0))
                lon = float(item.get('lon_num', 0))
                
                X_images.append(img_array)
                X_coords.append([lat, lon])
                y.append(self._normalize_year(item['Year']))
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        X_images = np.array(X_images)
        X_coords = np.array(X_coords)
        y = np.array(y)
        
        valid_mask = [y_val is not None for y_val in y]
        X_images = X_images[valid_mask]
        X_coords = X_coords[valid_mask]
        y = y[valid_mask]
        
        print(f"Prepared {len(X_images)} samples with image shape {X_images.shape} and coords shape {X_coords.shape}")
        return X_images, X_coords, y

    def _build_cnn_model(self, input_shape=(224, 224, 3)):
        """Build CNN model with DenseNet121 backbone + coordinate metadata"""
        print("Building CNN model with DenseNet121 backbone + coordinates...")
        print("Using transfer learning from ImageNet pre-trained weights...")
        
        # Image input branch - DenseNet121 as feature extractor
        image_input = Input(shape=input_shape, name='image_input')
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_tensor=image_input
        )
        
        # Transfer learning: freeze base layers, only fine-tune top layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        print(f"Frozen {len([l for l in base_model.layers if not l.trainable])} layers")
        print(f"Trainable {len([l for l in base_model.layers if l.trainable])} layers")
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Coordinate input branch
        coord_input = Input(shape=(2,), name='coord_input')
        coord_dense = Dense(16, activation='relu')(coord_input)
        coord_dense = BatchNormalization()(coord_dense)
        
        # Concatenate image features with coordinate features
        combined = Concatenate()([x, coord_dense])
        
        # Combined classification head for age regression
        combined = Dense(256, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.3)(combined)
        combined = Dense(128, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.2)(combined)
        
        # Sigmoid output for normalized year prediction
        age_output = Dense(1, activation='sigmoid', name='age_output')(combined)
        
        self.cnn_model = Model(inputs=[image_input, coord_input], outputs=age_output)
        
        self.cnn_model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['mae']
        )
        self.cnn_model.summary()

    def train_cnn(self, train_dataset, val_dataset=None, epochs=50, batch_size=32, image_size=(224, 224), use_augmentation=True):
        """Train CNN model with optional data augmentation"""
        # Prepare training data with augmentation
        X_train_imgs, X_train_coords, y_train = self._prepare_cnn_data(
            train_dataset, 
            image_size=image_size, 
            use_augmentation=use_augmentation
        )
        
        # Prepare validation data without augmentation
        if val_dataset is not None:
            X_val_imgs, X_val_coords, y_val = self._prepare_cnn_data(
                val_dataset, 
                image_size=image_size, 
                use_augmentation=False
            )
            validation_data = ([X_val_imgs, X_val_coords], y_val)
        else:
            validation_data = None
        
        # Build model
        self._build_cnn_model(input_shape=(image_size[0], image_size[1], 3))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        ]
        
        # Train
        aug_text = " with data augmentation" if use_augmentation else ""
        print(f"\nTraining CNN for {epochs} epochs{aug_text}...")
        self.history = self.cnn_model.fit(
            [X_train_imgs, X_train_coords],
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        print("Training complete.")

    def evaluate(self, test_dataset, image_size=(224, 224)):
        """Evaluate CNN model"""
        if self.cnn_model is None:
            print("No CNN model trained!")
            return None
        
        print("Evaluating CNN model...")
        X_test_imgs, X_test_coords, y_test_norm = self._prepare_cnn_data(
            test_dataset, 
            image_size=image_size, 
            use_augmentation=False  # Never augment test data
        )
        
        loss, mae_norm = self.cnn_model.evaluate([X_test_imgs, X_test_coords], y_test_norm, verbose=0)
        
        # Denormalize for actual MAE
        predictions_norm = self.cnn_model.predict([X_test_imgs, X_test_coords], verbose=0).flatten()
        predictions = self._denormalize_year(predictions_norm)
        y_test = self._denormalize_year(y_test_norm)
        
        mae_actual = np.mean(np.abs(y_test - predictions))
        
        print(f"\nTest Loss: {loss:.4f}")
        print(f"Test MAE: {mae_actual:.2f} years")
        
        print("\nExample Predictions:")
        for i in range(min(5, len(y_test))):
            print(f"True: {y_test[i]:.0f}, Predicted: {predictions[i]:.1f}, Error: {abs(y_test[i] - predictions[i]):.1f}")
        
        return mae_actual

    def save_model(self, path):
        """Save CNN model"""
        if self.cnn_model is not None:
            self.cnn_model.save(path)
            print(f"CNN model saved to {path}")

    def load_model(self, path):
        """Load CNN model
        """
        from pathlib import Path
        if not Path(path).exists():
            print(f"No cached model found at {path}")
            return False
        
        try:
            self.cnn_model = keras.models.load_model(path)
            print(f"CNN model loaded from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
