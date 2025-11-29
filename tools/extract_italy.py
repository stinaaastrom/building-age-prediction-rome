import json
from shapely.geometry import shape, Point

# CAUTION: the file italy_borders.geojson isn't committed in the repository due to
# its size. You can learn more about the origins of the file from the comment in the README.
# This script is here only for documentative purposes.
# Load the full countries GeoJSON
with open('italy_borders.geojson', 'r') as f:
    data = json.load(f)

# Find Italy - check multiple possible property names
italy_feature = None
for feature in data['features']:
    props = feature.get('properties', {})
    name = props.get('ADMIN') or props.get('name') or props.get('NAME') or props.get('admin')
    if name and 'Italy' in name:
        italy_feature = feature
        print(f"Found: {name}")
        break

if italy_feature:
    # Extract just Italy and save it
    italy_only = {
        "type": "FeatureCollection",
        "features": [italy_feature]
    }
    
    with open('italy_borders_only.geojson', 'w') as f:
        json.dump(italy_only, f)
    
    print("Italy borders extracted successfully!")
else:
    print("Italy not found in the dataset")
    # Print all country names to debug
    print("\nAvailable countries:")
    for feature in data['features'][:20]:
        props = feature.get('properties', {})
        name = props.get('ADMIN') or props.get('name') or props.get('NAME') or props.get('admin')
        print(f"  - {name}")
