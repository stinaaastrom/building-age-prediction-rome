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
import tensorflow as tf

class CNNModel:
    def __init__(self):
        # CNN model attributes
        self.image_processor = ImageProcessing()
        self.cnn_model = None
        self.history = None
        self.image_size = 224
        self.year_min = 1800.0
        self.year_max = 2020.0

        # Will be set during training
        self.year_mean = None
        self.year_std = None
        self.coord_mean = None  # np.array([lat_mean, lon_mean])
        self.coord_std = None   # np.array([lat_std, lon_std])

    def normalize_year(self, year):
        return (year - self.year_mean) / self.year_std

    def denormalize_year(self, year_norm):
        return year_norm * self.year_std + self.year_mean


    def prepare_data(self, dataset, image_size=(128, 128), use_augmentation=False, fit_stats=False, training=False):
        """Prepare images, coordinates, and labels for CNN with consistent normalization.

        - If fit_stats=True: compute and store year_mean/std and coord_mean/std from this dataset (use only for train).
        - If fit_stats=False: require that stats are already set (use for val/test) and reuse them.
        """
        X_images = []
        X_coords = []
        y = []

        years = np.array([item.get("Year") for item in dataset], dtype=float)
        # Fit year stats only once (train)
        if fit_stats or (self.year_mean is None or self.year_std is None):
            self.year_mean = float(np.nanmean(years))
            self.year_std = float(np.nanstd(years) if np.nanstd(years) > 1e-6 else 1.0)

        
        raw_coords = []
        for item in dataset:
            try:
                img = item['Picture']
                # Use CNN-specific processing (channel-last format)
                img_array = self.image_processor.image_processing_for_cnn(img, use_augmentation)
                
                # Extract coordinates
                lat = float(item.get('lat_num', 0)) if item.get('lat_num') is not None else 0.0
                lon = float(item.get('lon_num', 0)) if item.get('lon_num') is not None else 0.0
                
                X_images.append(img_array)
                raw_coords.append([lat, lon])
                y.append(self.normalize_year(item['Year']))
            except Exception as e:
                print(f"Error processing image: {e}")
                continue
        
        X_images = np.array(X_images)
        raw_coords = np.array(raw_coords, dtype=float)
        y = np.array(y)
        
        # Fit coord stats on train and standardize consistently
        if raw_coords.size > 0:
            if fit_stats or (self.coord_mean is None or self.coord_std is None):
                self.coord_mean = raw_coords.mean(axis=0)
                self.coord_std = raw_coords.std(axis=0)
                self.coord_std[self.coord_std < 1e-6] = 1.0
            X_coords = (raw_coords - self.coord_mean) / self.coord_std
        else:
            X_coords = raw_coords

        print(f"Prepared {len(X_images)} samples with image shape {X_images.shape} and coords shape {X_coords.shape}")
        return X_images, X_coords, y

    def predict_dataset(self, dataset):
        """
        Predicts years for a given dataset.
        Returns:
            y_pred: Predicted years
            y_true: Actual years
            coords: Coordinates (lat, lon)
        """
        X_images, X_coords, y_norm = self.prepare_cnn_data(dataset, fit_stats=False)
        
        # Predict
        y_pred_norm = self.cnn_model.predict([X_images, X_coords])
        y_pred = self.denormalize_year(y_pred_norm).flatten()
        
        # Denormalize y_true
        y_true = self.denormalize_year(y_norm)
        
        # Denormalize coords
        coords = X_coords * self.coord_std + self.coord_mean
        
        return y_pred, y_true, coords

    def build_cnn_model(self, input_shape=(224,224, 3)):
        """Build CNN model with DenseNet121 backbone + coordinate metadata"""
        # Image input branch - DenseNet121 as feature extractor
        image_input = Input(shape=input_shape, name="image_input")

        self.base_model = DenseNet121(
            weights="imagenet",
            include_top=False,
            input_tensor=image_input
        )

        # Freeze ALL DenseNet layers initially
        for layer in self.base_model.layers:
            layer.trainable = False

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)

        # ------------ COORDINATE INPUT (simple MLP) ------------
        coord_input = Input(shape=(2,), name="coord_input")

        c = Dense(16, activation="relu")(coord_input)
        c = Dense(8, activation="relu")(c)

        # ------------ COMBINE IMAGE + COORD FEATURES ------------
        combined = Concatenate()([x, c])

        # ------------ REGRESSION HEAD (simple) ------------
        r = Dense(128, activation="relu")(combined)
        r = Dropout(0.4)(r)

        r = Dense(64, activation="relu")(r)

        # Linear output for year regression
        output = Dense(1, activation="linear", name="year_output")(r)

        # ------------ BUILD & COMPILE MODEL ------------
        model = Model(
            inputs=[image_input, coord_input],
            outputs=output
        )

        model.compile(
            optimizer=Adam(learning_rate=1e-3),  # higher LR when backbone frozen
            loss="huber",
            metrics=["mae"]
        )

        self.cnn_model = model
        model.summary()

    def train_cnn(self, train_dataset, val_dataset=None, batch_size=16):
        print("Preparing data...")

        # Prepare training data (fit stats here)
        X_train_imgs, X_train_coords, y_train = self.prepare_data(train_dataset, use_augmentation=True, fit_stats=True)

        # Prepare validation data
        if val_dataset:
            X_val_imgs, X_val_coords, y_val = self.prepare_data(val_dataset, use_augmentation=False, fit_stats=False)
            validation = ([X_val_imgs, X_val_coords], y_val)
        else:
            validation = None

        # Build the model
        self.build_cnn_model(input_shape=(self.image_size, self.image_size, 3))


        print("Training...")
        self.cnn_model.fit(
            [X_train_imgs, X_train_coords],
            y_train,
            validation_data=validation,
            epochs=50,
            batch_size=batch_size,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)]
        )
        self.fine_tune_backbone(n_layers=25)

        self.cnn_model.fit(
            [X_train_imgs, X_train_coords],
            y_train,
            validation_data=validation,
            epochs=15,
            batch_size=batch_size,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
        )


    def fine_tune_backbone(self, n_layers=25):
        # Unfreeze last n layers EXCEPT BatchNorm
        for layer in self.base_model.layers[-n_layers:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

        self.cnn_model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss="huber",
            metrics=["mae"]
        )


    def evaluate(self, test_dataset, image_size=(224,224)):
        """Evaluate CNN model"""
        if self.cnn_model is None:
            print("No CNN model trained!")
            return None
        
        print("Evaluating CNN model...")
        X_test_imgs, X_test_coords, y_test_norm = self.prepare_data(
            test_dataset, 
            image_size=image_size, 
            use_augmentation=False,  # Never augment test data
            fit_stats=False
        )

        if len(X_test_imgs) == 0:
            print("No test samples available after preprocessing. Skipping evaluation.")
            return None
        
        loss, mae_norm = self.cnn_model.evaluate([X_test_imgs, X_test_coords], y_test_norm, verbose=0)
        
        # Denormalize for actual MAE
        predictions_norm = self.cnn_model.predict([X_test_imgs, X_test_coords], verbose=0).flatten()
        predictions = self.denormalize_year(predictions_norm)
        y_test = self.denormalize_year(y_test_norm)
        
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
