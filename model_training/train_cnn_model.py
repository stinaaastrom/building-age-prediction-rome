import numpy as np
from dataset_preparation.image_processing import ImageProcessing
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.applications import ResNet50
from sklearn.metrics import mean_absolute_error

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

    def _prepare_cnn_data(self, dataset, image_size=(128, 128)):
        """Prepare images and labels for CNN"""
        print(f"Preparing CNN data with image size {image_size}...")
        X = []
        y = []
        
        for item in dataset:
            try:
                img = item['Picture']
                img_array = self.image_processor.image_processing(img)
                
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

    def _build_cnn_model(self, input_shape=(224, 224, 3)):
        """Build CNN model with ResNet50 backbone"""
        print("Building CNN model with ResNet50 backbone...")
        
        # Use ResNet50 as feature extractor
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze early layers, only train last few
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Add custom layers on top
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        age_output = Dense(1, activation='sigmoid', name='age_output')(x)
        
        self.cnn_model = Model(inputs=base_model.input, outputs=age_output)
        
        self.cnn_model.compile(
            loss='mse',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['mae']
        )
        self.cnn_model.summary()

    def train_cnn(self, train_dataset, val_dataset=None, epochs=50, batch_size=32, image_size=(224, 224)):
        """Train using CNN model with data augmentation"""
        # Prepare data
        X_train, y_train = self._prepare_cnn_data(train_dataset, image_size=image_size)
        
        if val_dataset is not None:
            X_val, y_val = self._prepare_cnn_data(val_dataset, image_size=image_size)
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Build model
        self._build_cnn_model(input_shape=(image_size[0], image_size[1], 3))
        
        # Data augmentation for training
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
        ]
        
        # Train with augmentation
        print(f"\nTraining CNN for {epochs} epochs with data augmentation...")
        self.history = self.cnn_model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=validation_data,
            epochs=epochs,
            steps_per_epoch=len(X_train) // batch_size,
            callbacks=callbacks,
            verbose=1
        )
        print("Training complete.")

    def evaluate_cnn(self, test_dataset, image_size=(224, 224)):
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

    def save_cnn_model(self, path):
        """Save CNN model"""
        if self.cnn_model is not None:
            self.cnn_model.save(path)
            print(f"CNN model saved to {path}")

    def load_cnn_model(self, path):
        """Load CNN model"""
        self.cnn_model = keras.models.load_model(path)
        print(f"CNN model loaded from {path}")
