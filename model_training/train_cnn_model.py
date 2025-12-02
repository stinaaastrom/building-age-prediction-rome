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
        self.image_size = 320
        self.year_min = 1800.0
        self.year_max = 2020.0

    def normalize_year(self, year):
        return (year - self.year_mean) / self.year_std

    def denormalize_year(self, year_norm):
        return year_norm * self.year_std + self.year_mean


    def prepare_cnn_data(self, dataset, image_size=(128, 128), use_augmentation=False):
        """Prepare images, coordinates, and labels for CNN"""
        aug_text = " with augmentation" if use_augmentation else ""
        print(f"Preparing CNN data{aug_text} with image size {image_size}...")
        X_images = []
        X_coords = []
        y = []

        years = np.array([item["Year"] for item in dataset])
        self.year_mean = years.mean()
        self.year_std = years.std()

        
        for item in dataset:
            try:
                img = item['Picture']
                # Use CNN-specific processing (channel-last format)
                img_array = self.image_processor.image_processing_for_cnn(img, use_augmentation=False)
                
                # Extract coordinates
                lat = float(item.get('lat_num', 0))
                lon = float(item.get('lon_num', 0))
                
                X_images.append(img_array)
                X_coords.append([lat, lon])
                y.append(self.normalize_year(item['Year']))
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

    def build_cnn_model(self, input_shape=(320, 320, 3)):
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
        for layer in base_model.layers[:240]:
            layer.trainable = False
        
        print(f"Frozen {len([l for l in base_model.layers if not l.trainable])} layers")
        print(f"Trainable {len([l for l in base_model.layers if l.trainable])} layers")
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        
        # Coordinate input branch
        coord_input = Input(shape=(2,), name='coord_input')
        coord_dense = Dense(16, activation='relu')(coord_input)
        coord_dense = BatchNormalization()(coord_dense)
        
        # ------------ COORDINATE INPUT ------------
        coord_input = Input(shape=(2,), name="coord_input")
        c = Dense(32, activation="relu")(coord_input)
        c = BatchNormalization()(c)
        c = Dense(16, activation="relu")(c)
        c = BatchNormalization()(c)

        # ------------ COMBINE IMAGE + COORD FEATURES ------------
        combined = Concatenate()([x, c])

        # ------------ REGRESSION HEAD ------------
        r = Dense(256, activation="relu")(combined)
        r = BatchNormalization()(r)
        r = Dropout(0.3)(r)

        r = Dense(128, activation="relu")(r)
        r = BatchNormalization()(r)
        r = Dropout(0.2)(r)

        r = Dense(64, activation="relu")(r)

        # ⭐ VERY IMPORTANT: LINEAR OUTPUT FOR EXACT YEAR PREDICTION
        output = Dense(1, activation="linear", name="year_output")(r)

        model = Model(inputs=[image_input, coord_input], outputs=output)

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="huber",       # ⭐ BEST for year regression
            metrics=["mae"]
        )

        self.cnn_model = model
        model.summary()

    def train_cnn(self, train_dataset, val_dataset=None, epochs=60, batch_size=16):
        print("Preparing data...")

        # Prepare training data
        X_train_imgs, X_train_coords, y_train = self.prepare_cnn_data(train_dataset, use_augmentation=True)

        # Prepare validation data
        if val_dataset:
            X_val_imgs, X_val_coords, y_val = self.prepare_cnn_data(val_dataset, use_augmentation=False)
            validation = ([X_val_imgs, X_val_coords], y_val)
        else:
            validation = None

        # Build the model
        self.build_cnn_model(input_shape=(self.image_size, self.image_size, 3))

        callbacks = [
            EarlyStopping(patience=12, monitor="val_loss", restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=6, min_lr=1e-7)
        ]

        print("Training...")
        self.cnn_model.fit(
            [X_train_imgs, X_train_coords],
            y_train,
            validation_data=validation,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )

    def evaluate(self, test_dataset, image_size=(320, 320)):
        """Evaluate CNN model"""
        if self.cnn_model is None:
            print("No CNN model trained!")
            return None
        
        print("Evaluating CNN model...")
        X_test_imgs, X_test_coords, y_test_norm = self.prepare_cnn_data(
            test_dataset, 
            image_size=image_size, 
            use_augmentation=False  # Never augment test data
        )
        
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
