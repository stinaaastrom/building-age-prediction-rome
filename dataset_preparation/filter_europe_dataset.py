from datasets import load_dataset
import json
import urllib.request
from pathlib import Path
from shapely.geometry import shape, Point, mapping
from shapely.ops import unary_union

class EuropeDataset:
    def __init__(self, geojson_path: Path, dataset_name: str):
        self.dataset_name = dataset_name
        self.geojson_path = geojson_path
        self.europe_polygon = self._load_europe_borders()
        
    def _load_europe_borders(self):
        """Load Europe's geographic boundaries from GeoJSON"""
        with open(self.geojson_path, 'r') as f:
            data = json.load(f)
        
        # Convert GeoJSON to Shapely geometry
        europe_feature = data['features'][0]
        europe_polygon = shape(europe_feature['geometry'])
        print("Europe borders loaded successfully")
        return europe_polygon

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

    def _is_in_europe_polygon(self, lat_str, lon_str):
        """Check if point is within Europe's actual borders using Shapely"""
        # Only accept Northern Hemisphere (°N)
        if not (lat_str and '°N' in lat_str):
            return False
            
        # Handle Longitude: East or West
        is_east = lon_str and '°E' in lon_str
        is_west = lon_str and '°W' in lon_str
        
        if not (is_east or is_west):
            return False
            
        lat = self._parse_coordinate(lat_str, ['°N'])
        
        if is_east:
            lon = self._parse_coordinate(lon_str, ['°E'])
        else:
            lon = self._parse_coordinate(lon_str, ['°W'])
            if lon is not None:
                lon = -lon # West is negative
        
        if lat is None or lon is None:
            return False
        
        # Create point (lon, lat) - GeoJSON uses (longitude, latitude) order
        point = Point(lon, lat)
        
        # Check if point is within Europe's borders
        return self.europe_polygon.contains(point)

    def _filter_condition(self, example):
        # Extract fields
        lat_str = example.get('Latitude', '')
        lon_str = example.get('Longitude', '')
        
        # Condition: Geographic location within Europe
        if self._is_in_europe_polygon(lat_str, lon_str):
            return True
        
        return False

    def get_filtered_dataset(self, split="train"):
        print(f"Loading dataset '{self.dataset_name}' split '{split}'...")
        dataset = load_dataset(self.dataset_name, split=split)
        
        print(f"Filtering dataset for Europe (split={split})...")
        filtered_dataset = dataset.filter(self._filter_condition, num_proc=4)
        
        print(f"Original size: {len(dataset)}, Filtered size: {len(filtered_dataset)}")
        return filtered_dataset
