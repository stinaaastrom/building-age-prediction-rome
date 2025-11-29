# building-age-prediction-rome

## Data: Italy borders GeoJSON

The file `italy_borders.geojson` in this repository was extracted from the public
countries GeoJSON maintained in the `datasets/geo-countries` GitHub repository.

Source URL used to obtain the full countries GeoJSON:

```
https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson
```

This GeoJSON is used by `filter_Italy_dataset.py` (via Shapely) to perform precise
point-in-polygon checks to determine whether image coordinates fall within Italy's
official land boundaries.

The visual of the Italian borders is taken from a geojson.io render based on the extracted
polygon edges.
