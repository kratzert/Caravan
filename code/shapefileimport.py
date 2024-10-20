import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

# Load the existing shapefile
existing_shapefile = gpd.read_file(r"C:\Users\amixg\Downloads\dati_dbtr")

# Create a new GeoDataFrame for Emilia-Romagna
emilia_romagna_data = gpd.GeoDataFrame(
    {
        "region": ["Emilia-Romagna"],
        "flood_prev_measure": ["Describe the measure here"],
        "geometry": [Polygon([(9.20, 43.75), (12.75, 43.75), (12.75, 45.15), (9.20, 45.15)])]
    }
)

# Ensure the CRS matches your existing shapefile
emilia_romagna_data.set_crs(existing_shapefile.crs, inplace=True)

# Concatenate the new data to the existing shapefile
updated_shapefile = gpd.GeoDataFrame(pd.concat([existing_shapefile, emilia_romagna_data], ignore_index=True))

# Save the updated data as a GeoPackage
updated_shapefile.to_file(r"C:\Users\amixg\Downloads\updated_data.gpkg", driver="GPKG")


# Replace with the path to your .gpkg file
gdf = gpd.read_file("C:/Users/amixg/Downloads/updated_data.gpkg")

# Now you can work with the data in the GeoDataFrame
print(gdf.head())