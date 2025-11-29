# building-age-prediction-rome

## Data: Italy borders GeoJSON

The file `italy_borders_only.geojson` in this repository was extracted from the public
countries GeoJSON maintained in the `datasets/geo-countries` GitHub repository.

Source URL used to obtain the full countries GeoJSON:

```
https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson
```

I used the small helper script `extract_italy.py` to locate the feature with the
country name "Italy" and write a new file containing only that feature. The script
and `italy_borders_only.geojson` are included in this repository.

This GeoJSON is used by `filter_Italy_dataset.py` (via Shapely) to perform precise
point-in-polygon checks to determine whether image coordinates fall within Italy's
official land boundaries.
