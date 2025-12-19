# Building Age Prediction - Rome

This project aims to predict the construction year of buildings in Rome using images and geographical data. It employs various machine learning approaches, including Convolutional Neural Networks (CNN), Support Vector Machines (SVM), and Gradient Boosting Models (GBM).

## Project Structure

The repository is organized modularly:

*   **`main.py`**: The main entry point for training and evaluating the models. This is where K-Fold Cross-Validation is controlled.
*   **`dataset_preparation/`**: Scripts for preparing, filtering, and processing the dataset.
    *   `image_processing.py`: Image preprocessing for CNNs and other models.
    *   `filter_italy_dataset.py`: Filters relevant data from the complete dataset.
*   **`model_training/`**: Contains the training logic for the different models.
    *   `train_cnn_model.py`: Training the CNN (based on DenseNet/EfficientNet).
    *   `train_gradient_boosting_model.py`: Training the GBM.
    *   `train_svr_model.py`: Training the Support Vector Regression.
    *   Also contains saved models (`.keras`, `.joblib`).
*   **`result_visualization/`**: Tools for analyzing and visualizing the results.
    *   `visualize_predictions.py`: Shows predictions compared to ground truth.
    *   `visualize_errors_geographically.py`: Maps prediction errors on a map.
    *   `find_worst_predictions.py`: Identifies the largest outliers.
*   **`dataset_exploration/`**: SQL queries and GeoJSON files for data exploration.
*   **`resources/`**: Geographical borders and helper files (e.g., `italy_borders.geojson`).

## Installation

1.  Clone the repository.
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Mac/Linux
    # or
    .\venv\Scripts\activate   # Windows
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start training and evaluation, run `main.py`:

```bash
python main.py
```

In `main.py`, various parameters can be configured, such as the model type to use (`gbm`, `cnn`, `svr`) or the number of folds for cross-validation.

## Models

The project currently supports the following model architectures:
*   **CNN**: Deep Learning approach for direct image analysis.
*   **SVR**: Support Vector Regression on extracted features.
*   **GBM**: Gradient Boosting for tabular/feature-based predictions.
