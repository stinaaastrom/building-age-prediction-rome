from datasets import load_dataset
import json
from pathlib import Path
from shapely.geometry import shape, Point

class ItalyDataset:
    def __init__(self, geojson_path: Path, dataset_name: str):
        self.dataset_name = dataset_name
        self.geojson_path = geojson_path
        self.italy_polygon = self._load_italy_borders()

    def _load_italy_borders(self):
        """Load Italy's geographic boundaries from GeoJSON"""
        with open(self.geojson_path, 'r') as f:
            data = json.load(f)
        
        # Convert GeoJSON to Shapely geometry
        italy_feature = data['features'][0]
        italy_polygon = shape(italy_feature['geometry'])
        print("Italy borders loaded successfully")
        return italy_polygon

    def _parse_coordinate(self, coord_str, hemisphere_indicators):
        """Parse coordinate string and return float value or None"""
        if not coord_str:
            return None
        
        clean_str = coord_str
        for indicator in hemisphere_indicators:
            clean_str = clean_str.replace(indicator, '')
        
        try:
            return float(clean_str)
        except ValueError:
            return None

    def _is_in_italy_polygon(self, lat_str, lon_str):
        """Check if point is within Italy's actual borders using Shapely"""
        # Only accept Northern Hemisphere (°N) and Eastern Hemisphere (°E)
        if not (lat_str and '°N' in lat_str):
            return False
        if not (lon_str and '°E' in lon_str):
            return False
        
        lat = self._parse_coordinate(lat_str, ['°N'])
        lon = self._parse_coordinate(lon_str, ['°E'])
        
        if lat is None or lon is None:
            return False
        
        # Create point (lon, lat) - GeoJSON uses (longitude, latitude) order
        point = Point(lon, lat)
        
        # Check if point is within Italy's borders
        return self.italy_polygon.contains(point)

    def _filter_condition(self, example):
        # Extract fields
        country = example.get('Country', '')
        lat_str = example.get('Latitude', '')
        lon_str = example.get('Longitude', '')
        
        # Condition 1: Country is explicitly 'Italy'
        if country == 'Italy':
            return True
        
        # Condition 2: Geographic location within Italy
        if self._is_in_italy_polygon(lat_str, lon_str):
            return True
        
        return False

    def get_filtered_dataset(self, split="train"):
        print(f"Loading dataset '{self.dataset_name}' split '{split}'...")
        dataset = load_dataset(self.dataset_name, split=split)
        
        print(f"Filtering dataset for Italy (split={split})...")
        filtered_dataset = dataset.filter(self._filter_condition, num_proc=4)
        
        print(f"Original size: {len(dataset)}, Filtered size: {len(filtered_dataset)}")
        return filtered_dataset
