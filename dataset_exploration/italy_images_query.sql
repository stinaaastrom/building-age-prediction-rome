-- SQL query to filter images from Italy
-- Assumption: Table name is 'train'.
-- Coordinates for Italy (rough bounding box):
-- Latitude: approx. 36.8 to 47.1 (North)
-- Longitude: approx. 6.6 to 18.6 (East)

SELECT COUNT(*)
FROM train
WHERE 
        Country = 'Italy' 
        OR 
        (
        -- Ensure Northern Hemisphere (Italy is North)
        Latitude LIKE '%°N' 
        AND
        -- Ensure Eastern Hemisphere (Italy is East)
        Longitude LIKE '%°E'
        AND
        (
            -- Define Italy using multiple smaller bounding boxes for better accuracy
            
            -- 1. Northern Italy (Wide: Alps, Milan, Venice)
            (
                CAST(REPLACE(Latitude, '°N', '') AS DECIMAL(10, 6)) BETWEEN 43.5 AND 47.1
                AND 
                CAST(REPLACE(Longitude, '°E', '') AS DECIMAL(10, 6)) BETWEEN 6.6 AND 14.0
            )
            OR
            -- 2. Central Italy (Tuscany, Rome) - Excludes Corsica (West) and Croatia (East)
            (
                CAST(REPLACE(Latitude, '°N', '') AS DECIMAL(10, 6)) BETWEEN 41.5 AND 43.5
                AND 
                CAST(REPLACE(Longitude, '°E', '') AS DECIMAL(10, 6)) BETWEEN 9.8 AND 16.2
            )
            OR
            -- 3. Southern Italy (Naples, Puglia, Calabria) - Shifted East
            (
                CAST(REPLACE(Latitude, '°N', '') AS DECIMAL(10, 6)) BETWEEN 37.9 AND 41.5
                AND 
                CAST(REPLACE(Longitude, '°E', '') AS DECIMAL(10, 6)) BETWEEN 13.0 AND 18.6
            )
            OR
            -- 4. Sicily
            (
                CAST(REPLACE(Latitude, '°N', '') AS DECIMAL(10, 6)) BETWEEN 36.6 AND 38.3
                AND 
                CAST(REPLACE(Longitude, '°E', '') AS DECIMAL(10, 6)) BETWEEN 12.4 AND 15.7
            )
            OR
            -- 5. Sardinia
            (
                CAST(REPLACE(Latitude, '°N', '') AS DECIMAL(10, 6)) BETWEEN 38.8 AND 41.4
                AND 
                CAST(REPLACE(Longitude, '°E', '') AS DECIMAL(10, 6)) BETWEEN 8.1 AND 9.8
            )
        )
    )
    ;
