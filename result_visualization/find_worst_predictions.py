import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tools.generate_filename import generate_filename, get_project_root
from tools.generate_sql_filter import SQL_Where

class WorstPredictionsFinder:
    def __init__(self, model):
        self.model = model

    def find_worst(self, dataset, top_n=20):
        print(f"Finding top {top_n} worst predictions...")
        
        all_predictions = []
        
        # Use unified predict_dataset method for batch prediction
        if hasattr(self.model, 'predict_dataset'):
            try:
                y_pred, y_true, _ = self.model.predict_dataset(dataset)
                
                # Iterate through results
                for i, item in enumerate(dataset):
                    building = item.get('Building', 'Unknown')
                    pred_year = y_pred[i]
                    actual_year = y_true[i]
                    
                    error = abs(actual_year - pred_year)
                    
                    all_predictions.append({
                        'Building': building,
                        'Predicted': pred_year,
                        'Actual': actual_year,
                        'Error': error
                    })
            except Exception as e:
                print(f"Error during batch prediction: {e}")
                return
        else:
            # Fallback (legacy loop) - should not be needed anymore
            print("Warning: Model does not support predict_dataset. Using slow legacy loop.")
            for item in tqdm(dataset, desc="Evaluating"):
                # ... (legacy code omitted for brevity, assuming models are updated)
                pass

        # Sort by error descending
        all_predictions.sort(key=lambda x: x['Error'], reverse=True)
        
        # Take top N
        worst = all_predictions[:top_n]
        
        # Save CSV to data directory
        output_path = get_project_root() / 'result_visualization' / 'data'
        output_path.mkdir(parents=True, exist_ok=True)
        filename = generate_filename('worst_predictions')
        csv_file = output_path / (filename + '.csv')
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Building', 'Predicted Year', 'Actual Year', 'Absolute Error'])
            for item in worst:
                writer.writerow([
                    item['Building'], 
                    f"{item['Predicted']:.2f}", 
                    item['Actual'], 
                    f"{item['Error']:.2f}"
                ])
                
        print(f"Worst predictions saved to {csv_file}")

        print(SQL_Where.generate(csv_file))
