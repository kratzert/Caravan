import geopandas as gpd
from shapely.geometry import Polygon

# Load the existing shapefile
existing_shapefile = gpd.read_file("path_to_your_existing_shapefile.shp")

# Create a new GeoDataFrame for Emilia-Romagna
emilia_romagna_data = gpd.GeoDataFrame(
    {
        "region": ["Emilia-Romagna"],
        "flood_prevention_measure": ["Describe the measure here"],
        "geometry": [Polygon([(9.20, 43.75), (12.75, 43.75), (12.75, 45.15), (9.20, 45.15)])]
    }
)

# Ensure the CRS matches your existing shapefile
emilia_romagna_data.crs = existing_shapefile.crs

# Append the new data to the existing shapefile
updated_shapefile = existing_shapefile.append(emilia_romagna_data, ignore_index=True)

# Save the updated shapefile
updated_shapefile.to_file("path_to_save_updated_shapefile.shp")