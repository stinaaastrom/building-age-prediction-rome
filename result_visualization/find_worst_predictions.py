import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tools.generate_filename import Filename 
from tools.generate_sql_filter import SQL_Where

class WorstPredictionsFinder:
    def __init__(self, model):
        self.model = model

    def find_worst(self, dataset, output_path: Path, top_n=20):
        print(f"Finding top {top_n} worst predictions...")
        
        all_predictions = []
        
        # Iterate over the dataset
        # We process one by one to ensure alignment between metadata and predictions
        # (Batch processing would require handling potential dropped images in feature extraction)
        for item in tqdm(dataset, desc="Evaluating"):
            img = item['Picture']
            year_true = item['Year']
            building = item['Building']
            
            if year_true is None:
                continue
                
            # Create a mini batch of 1 for the model
            # The model expects a dict with 'Picture' list
            mini_batch = {'Picture': [img]}
            
            try:
                extracted = self.model._extract_features_batch(mini_batch)
                features = extracted['features']
                
                if len(features) == 0:
                    continue
                    
                # Predict
                # features is (1, 2048)
                pred_year = self.model.svr.predict(features)[0]
                error = abs(year_true - pred_year)
                
                all_predictions.append({
                    'Building': building,
                    'Predicted': pred_year,
                    'Actual': year_true,
                    'Error': error
                })
            except Exception as e:
                print(f"Error processing {building}: {e}")
                continue

        # Sort by error descending
        all_predictions.sort(key=lambda x: x['Error'], reverse=True)
        
        # Take top N
        worst = all_predictions[:top_n]
        
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        filename = Filename.generate('worst_predictions')
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
