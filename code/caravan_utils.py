# This file is part of Caravan project/dataset. See https://github.com/kratzert/Caravan for details.
#
# You should have received a copy of the BSD-3-Clause license along with this file. If not,
# see https://opensource.org/licenses/BSD-3-Clause.

from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from pytz import timezone, utc
from typing import Dict, List

import numpy as np
import pandas as pd
import xarray
from numba import njit
from timezonefinder import TimezoneFinder
from tqdm.notebook import tqdm

# Keep these dates. They are the original time period used to compute climate indices. Changing these dates would
# remove the compareability between extensions, if longer ERA5-Land periods are downloaded.
_CARAVAN_START_DATE = pd.to_datetime('1981-01-01', format="%Y-%m-%d")
_CARAVAN_END_DATE = pd.to_datetime('2020-12-31', format="%Y-%m-%d")


def process_earth_engine_outputs(csv_files: List[Path],
                                 basin_id_field: str,
                                 era5l_bands: List[str],
                                 output_dir: Path,
                                 num_workers: int,
                                 basin_prefix: str | None = None) -> List[pd.DataFrame]:
    """Processes Earth Engine output files in parallel into per-basin netCDF files.
    
    Parameters
    ----------
    csv_files : List[Path]
        List of Path objects pointing to the csv files that were produced by Earth Engine as a result of the first 
        notebook.
    basin_id_field : str
        Name of the attribute field that corresponds to the unique basin id. Same name as in the shape file that was
        used to derive the spatially averaged forcing data.
    era5l_bands : List[str]
        Name of ERA5-Land bands that should be processed.
    output_dir : Path
        Directory in which the processed per-basin netCDF files are stored.
    num_workers : int
        Number of parallel workers. Usually this should be a value lower than the maximum number of cores available on 
        your system.
    basin_prefix : str | None
        String prefix that is prepended to the gauge ids in the basin_id_field in the following format {prefix}_{id}.
    """

    # Split Earth Engine outputs into per-basin netCDF files.
    with Pool(num_workers) as p:
        _ = list(
            tqdm(p.imap(
                partial(_process_single_file,
                        basin_id_field=basin_id_field,
                        era5l_bands=era5l_bands,
                        output_dir=output_dir,
                        basin_prefix=basin_prefix), csv_files),
                 total=len(csv_files),
                 desc="Splitting Earth Engine output into per-basin files."))

    # Stack the per-basin netCDF file to one (combined) file.
    basin_dirs = list((output_dir / "temp").glob('*'))

    # Split Earth Engine outputs into per-basin netCDF files.
    with Pool(num_workers) as p:
        _ = list(
            tqdm(p.imap(stack_per_basin_netcdfs, basin_dirs),
                 total=len(basin_dirs),
                 desc="Combining files per-basin into one file."))


def _process_single_file(csv_file: Path, basin_id_field: str, era5l_bands: List[str], output_dir: Path,
                         basin_prefix: str | None):
    # Load all data of one csv file into memory.
    df = load_and_clean_csv_file(csv_file=csv_file,
                                 basin_id_field=basin_id_field,
                                 era5l_bands=era5l_bands,
                                 basin_prefix=basin_prefix)

    # Get unique list of basin ids. The field is renamed in the function above to gauge_id.
    basins = list(set(df["gauge_id"].to_list()))

    # Loop over basins and create per-basin data and store to temp files.
    for basin in basins:
        try:
            split_by_basin_and_save(df=df, basin=basin, output_dir=output_dir, filename=csv_file.stem)
        except:
            print(csv_file, basin)


def load_and_clean_csv_file(csv_file: Path, basin_id_field: str, era5l_bands: List[str],
                            basin_prefix: str | None) -> pd.DataFrame:
    """Load raw Earth Engine outputs and convert into time indexed DataFrame.
    
    Parameters
    ----------
    csv_file : Path
        Path object pointing to a csv file that were produced by Earth Engine as a result of the first notebook.
    basin_id_field : str
        Name of the attribute field that corresponds to the unique basin id. Same name as in the shape file that was
        used to derive the spatially averaged forcing data.
    era5l_bands : List[str]
        Name of ERA5-Land bands that should be processed.
    basin_prefix : str
        String prefix that is prepended to the gauge ids in the basin_id_field in the following format {prefix}_{id}.

    Returns
    -------
    pd.DataFrame
        Time-indexed DataFrame that contains the hourly ERA5-Land data of all basins.
    """
    # load raw Earth Engine output as DataFrame
    df = pd.read_csv(csv_file, dtype={basin_id_field: str})
    # Unify the naming of the basin id column
    df = df.rename(columns={basin_id_field: "gauge_id"})

    if basin_prefix is not None and basin_prefix:
        df["gauge_id"] = df["gauge_id"].map(lambda x: f"{basin_prefix}_{x}")

    # Create datetime column
    df["date_str"] = df["system:index"].map(lambda x: x[:11].replace('T', '_'))
    df["date"] = pd.to_datetime(df["date_str"], format='%Y%m%d_%H')

    # remove unnecessary columns
    drop_cols = [c for c in df.columns if c not in era5l_bands + ['gauge_id', 'date']]
    df = df.drop(drop_cols, axis=1)

    # set datetime column as index
    df = df.set_index('date')

    return df


def split_by_basin_and_save(df: pd.DataFrame, basin: str, output_dir: Path, filename: str):
    """Extracts basin data from DataFrame and stores data as netCDF file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the combined ERA5-Land data of all basins.
    basin : str
        Unique basin id.
    output_dir : Path
        Directory in which the processed per-basin netCDF files are stored.
    """
    df_basin = df.loc[df['gauge_id'] == basin]
    df_basin = df_basin.drop(["gauge_id"], axis=1)

    # store as netCDF and float32 to save disk space
    xr_basin = xarray.Dataset.from_dataframe(df_basin).astype(np.float32)
    output_path = output_dir / "temp" / basin / f"{filename}.nc"
    if not output_path.parent.is_dir():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    xr_basin.to_netcdf(output_path)


def stack_per_basin_netcdfs(basin_dir: Path):
    """Combines multiple netCDF files of a single basin into one object.
    
    Parameters
    ----------
    basin_dir : Path
        Directory that contains individual netCDF files for one basin. All netCDF files in this directory are loaded
        and concatenated in time and then saved in the same directory in a file called `combined.nc`.
    """
    xrs = []
    batch_files = list(basin_dir.glob('*.nc'))

    # Load per-basin batch files.
    for filepath in batch_files:
        xrs.append(xarray.open_dataset(filepath))

    # Stack all data and sort by time index.
    if xrs:
        xr = xarray.concat(xrs, dim="date")
        xr = xr.sortby('date', ascending=True)
        xr.to_netcdf(basin_dir / 'combined.nc')
    else:
        raise ValueError(f"Could not find any .nc files in {basin_dir}")


def aggregate_df_to_daily(df: pd.DataFrame,
                          gauge_lat: float,
                          gauge_lon: float,
                          mean_vars: List[str] = None,
                          min_vars: List[str] = None,
                          max_vars: List[str] = None,
                          sum_vars: List[str] = None):
    """Aggregates hourly ERA5-Land data in UTC to daily data in local time.

    Parameters
    ----------
    df : pd.DataFrame
        Timeindexed DataFrame in hourly resolution, containing the raw ERA5-Land data of one basin.
    gauge_lat : float
        Latitude (WGS 84) of the streamflow gauge. Used to determine local timezone.
    gauge_lon : float
        Longitude (WGS 84) of the streamflow gauge. Used to determine the local timezone.
    mean_vars : List[str], optional
        List of DataFrame columns that are aggregated to daily resolution by computing the daily mean. For each name in
        this list, a new column is added with the column name followed by the suffix '_mean'.
    min_vars : List[str], optional
        List of DataFrame columns that are aggregated to daily resolution by computing the daily min. For each name in
        this list, a new column is added with the column name followed by the suffix '_min'.
    max_vars : List[str], optional
        List of DataFrame columns that are aggregated to daily resolution by computing the daily max. For each name in
        this list, a new column is added with the column name followed by the suffix '_max'.
    sum_vars : List[str], optional
        List of DataFrame columns that are aggregated to daily resolution by computing the daily sum. For each name in
        this list, a new column is added with the column name followed by the suffix '_sum'.
    
    Returns
    -------
    pd.DataFrame
        Timeindexed DataFrame in daily resolution (local time).
    """

    if all([not x for x in [mean_vars, min_vars, max_vars, sum_vars]]):
        raise ValueError("You need to pass at least one of mean_vars, min_vars, max_vars, and sum_vars")

    df = _utc_to_local_standard_time(df.copy(), lat=gauge_lat, lon=gauge_lon)

    # start and end date of forcing data where we have all 24 data points of one day
    start_date = df.loc[df.index.hour == 1].first_valid_index()
    end_date = df.loc[df.index.hour == 0].last_valid_index()
    df = df[start_date:end_date]

    dfs = []
    if mean_vars:
        df_mean = df[mean_vars].resample('1D', offset=pd.Timedelta(hours=1)).mean()
        df_mean = df_mean.rename(columns={col: f"{col}_mean" for col in df_mean.columns})
        dfs.append(df_mean)

    if min_vars:
        df_min = df[min_vars].resample('1D', offset=pd.Timedelta(hours=1)).min()
        df_min = df_min.rename(columns={col: f"{col}_min" for col in df_min.columns})
        dfs.append(df_min)

    if max_vars:
        df_max = df[max_vars].resample('1D', offset=pd.Timedelta(hours=1)).max()
        df_max = df_max.rename(columns={col: f"{col}_max" for col in df_max.columns})
        dfs.append(df_max)

    if sum_vars:
        df_sum = df[sum_vars].resample('1D', offset=pd.Timedelta(hours=1)).sum()
        df_sum = df_sum.rename(columns={col: f"{col}_sum" for col in df_sum.columns})
        dfs.append(df_sum)

    aggregates = pd.concat(dfs, axis=1)
    aggregates.index = aggregates.index.strftime('%Y-%m-%d')
    aggregates.index = pd.to_datetime(aggregates.index, format="%Y-%m-%d")
    return aggregates


def calculate_climate_indices(df: pd.DataFrame,
                              period_start_date: pd.Timestamp = _CARAVAN_START_DATE,
                              period_end_date: pd.Timestamp = _CARAVAN_END_DATE) -> Dict[str, float]:
    """Calculates various climate indices from ERA5-Land bands.
    
    See Caravan publication for details.

    Parameters
    ----------
    df : pd.DataFrame
        Timeindexed DataFrame in daily resolution. Must contain the columns 'total_precipitation_sum', 
        'potential_evaporation_sum' and 'temperature_2m_mean'.
    period_start_date : pd.Timestamp
        Can be used to define a different start date of the timeseries that should be considered for 
        computing the climate indices. Note, if you plan to release a Caravan extension, please use
        the default start date, i.e. don't pass a period_start_date to this function.
    period_end_date : pd.Timestamp
        Can be used to define a different end date of the timeseries that should be considered for 
        computing the climate indices. Note, if you plan to release a Caravan extension, please use
        the default end date, i.e. don't pass a period_end_date to this function.

    Returns
    -------
    Dict[str, float]
        Dictionary, where each key-value-pair is the name of one climate index and the corresponding value.
    """
    required_columns = [
        'total_precipitation_sum',
        'potential_evaporation_sum_ERA5_LAND',
        'potential_evaporation_sum_FAO_PENMAN_MONTEITH',
        'temperature_2m_mean',
    ]
    if any([x not in df.columns for x in required_columns]):
        raise RuntimeError(f"DataFrame is missing one of {required_columns} as column")

    # Before computing any index, we make sure to slice to the original Caravan periods
    df = df.loc[slice(period_start_date, period_end_date)]

    # Mean daily precip
    p_mean = df["total_precipitation_sum"].mean()
    # Mean daily PET
    pet_mean_era5 = df["potential_evaporation_sum_ERA5_LAND"].mean()
    pet_mean_fao = df["potential_evaporation_sum_FAO_PENMAN_MONTEITH"].mean()

    # Aridity index
    aridity_era5 = pet_mean_era5 / p_mean
    aridity_fao = pet_mean_fao / p_mean

    # Compute moistuer and seasonality index once with ERA5 PET and once with FAO PM PET
    annual_moisture_index_era5, seasonality_era5 = _get_moisture_and_seasonality_index(
        precipitation=df["total_precipitation_sum"], pet=df["potential_evaporation_sum_ERA5_LAND"])
    annual_moisture_index_fao, seasonality_fao = _get_moisture_and_seasonality_index(
        precipitation=df["total_precipitation_sum"], pet=df["potential_evaporation_sum_FAO_PENMAN_MONTEITH"])

    # Fraction of mean monthly precipipitation falling as snow (see Knoben)
    mean_monthly_precip = df["total_precipitation_sum"].groupby(df.index.month).mean()
    mean_monthly_temp = df["temperature_2m_mean"].groupby(df.index.month).mean()
    frac_snow = mean_monthly_precip.loc[mean_monthly_temp < 0].sum() / mean_monthly_precip.sum()

    high_prec_freq = len(df.loc[df["total_precipitation_sum"] >= 5 * p_mean]) / len(df)
    low_prec_freq = len(df.loc[df["total_precipitation_sum"] < 1]) / len(df)

    precip = df["total_precipitation_sum"].values
    idx = np.where(precip < 1)[0]
    groups = _split_list(idx)
    if groups:
        low_precip_dur = np.mean(np.array([len(p) for p in groups]))
    else:
        low_precip_dur = 0.0

    idx = np.where(precip >= 5 * p_mean)[0]
    groups = _split_list(idx)
    if groups:
        high_prec_dur = np.mean(np.array([len(p) for p in groups]))
    else:
        high_prec_dur = 0.0

    climate_indices = {
        'p_mean': p_mean,
        'pet_mean_ERA5_LAND': pet_mean_era5,
        'pet_mean_FAO_PM': pet_mean_fao,
        'aridity_ERA5_LAND': aridity_era5,
        'aridity_FAO_PM': aridity_fao,
        'frac_snow': frac_snow,
        'moisture_index_ERA5_LAND': annual_moisture_index_era5,
        'seasonality_ERA5_LAND': seasonality_era5,
        'moisture_index_FAO_PM': annual_moisture_index_fao,
        'seasonality_FAO_PM': seasonality_fao,
        'high_prec_freq': high_prec_freq,
        'high_prec_dur': high_prec_dur,
        'low_prec_freq': low_prec_freq,
        'low_prec_dur': low_precip_dur
    }

    return climate_indices


def _get_moisture_and_seasonality_index(precipitation, pet) -> tuple[float, float]:

    mean_monthly_precip = precipitation.groupby(precipitation.index.month).mean()
    mean_monthly_pet = pet.groupby(pet.index.month).mean()

    # Average annual moisture index (see Knoben)
    p_gt_et = 1 - mean_monthly_pet.loc[mean_monthly_precip > mean_monthly_pet] / mean_monthly_precip.loc[
        mean_monthly_precip > mean_monthly_pet]
    srs = pd.Series(np.zeros((12), dtype=np.float32), index=mean_monthly_pet.index, name='dummy')
    p_eq_et = srs.loc[mean_monthly_precip == mean_monthly_pet]
    p_lt_et = mean_monthly_precip.loc[mean_monthly_precip < mean_monthly_pet] / mean_monthly_pet.loc[
        mean_monthly_precip < mean_monthly_pet] - 1
    monthly_moisture_index = pd.concat([p_gt_et, p_eq_et, p_lt_et])

    annual_moisture_index = monthly_moisture_index.mean()

    # Seasonality (see Knoben)
    seasonality = monthly_moisture_index.max() - monthly_moisture_index.min()

    return annual_moisture_index, seasonality


def disaggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Disaggregate daily accumulated features into hourly 'instantaneous' values.
    
    The following features in ERA5-Land are accumulates over the day (UTC): 'total_precipitation', 
    'surface_net_solar_radiation', 'surface_net_thermal_radiation', and 'potential_evaporation'. This function 
    disaggregates these features into hourly 'instantaneous' values instead.
    
    Parameters
    ----------
    df : pd.DataFrame
        Timeindexed DataFrame in hourly resolution (UTC), containing the raw ERA5-Land data of one basin.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same columns but with all columns being instantaneous values.
    """
    # List of columns that are accumulated in ERA5Land
    columns = [
        "total_precipitation", "surface_net_solar_radiation", "surface_net_thermal_radiation", "potential_evaporation"
    ]

    # sanity check which features exist in df
    columns = [c for c in columns if c in df.columns]

    # Calculate the difference between two time steps
    temp = df[columns].diff(1)

    # replace every 00:00 to 01:00 value with the original data
    temp.loc[temp.index.hour == 1] = df[columns].loc[df.index.hour == 1].values

    # the first time step in diff time series is NaN, replace with orignal data
    temp.iloc[0] = df[columns].iloc[0]

    # overwrite data
    df[columns] = temp

    return df


def era5l_unit_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ERA5L units to commonly used units in hydrology
    
    Parameters
    ----------
    df : pd.DataFrame
        Timeindexed DataFrame in daily resolution.

    Returns
    -------
    pd.DataFrame
        DataFrame with unit-converted data. Check `get_metadata_info()` for details about the output units.
    """

    for col in df.columns:
        if col == "dewpoint_temperature_2m":
            # Kelvin -> Celsius
            df[col] = df[col] - 273.15

        elif col == "temperature_2m":
            # Kelvin -> Celsius
            df[col] = df[col] - 273.15

        elif col == "snow_depth_water_equivalent":
            # m -> mm
            df[col] = df[col] * 1000

        elif col == "surface_net_solar_radiation":
            # J/m2 -> W/m2
            df[col] = df[col] / 3600

        elif col == "surface_net_thermal_radiation":
            # J/m2 -> W/m2
            df[col] = df[col] / 3600

        elif col == "surface_pressure":
            # Pa -> kPa
            df[col] = df[col] / 1000

        elif col == "total_precipitation":
            # m -> mm
            df[col] = df[col] * 1000

        elif col == "potential_evaporation":
            # m -> mm
            df[col] = df[col] * 1000

    return df


def get_metadata_info(xr: xarray.Dataset) -> Dict[str, str]:
    """Compile unit metadata depending on the included ERA5-Land features.
    
    Parameters
    ----------
    xr : xarray.Dataset
        Dataset with (aggregates of) ERA5-Land features.

    Returns
    -------
    Dict[str, str]
        Dictionary containing one key-value pair for each ERA5-Land feature that is available in `xr`. The key 
        corresponds to the band-name of ERA5-Land (not the aggregate with a suffix) and the value is the feature name
        and the unit (as converted by `era5l_unit_conversion`, not the original ERA5-Land units).
    """
    # list of available features
    features = list(xr.variables)

    metadata = {}

    for feature in features:

        if feature.startswith("temperature_2m"):
            metadata["temperature_2m"] = "2m air temperature [°C]"

        elif feature.startswith("snow_depth_water_equivalent"):
            metadata["snow_depth_water_equivalent"] = "ERA5-Land Snow-Water-Equivalent [mm]"

        elif feature.startswith("surface_net_solar_radiation"):
            metadata["surface_net_solar_radiation"] = "Surface net solar radiation [W/m2]"

        elif feature.startswith("surface_net_thermal_radiation"):
            metadata["surface_net_thermal_radiation"] = "Surface net thermal radiation [W/m2]"

        elif feature.startswith("surface_pressure"):
            metadata["surface_pressure"] = "Surface pressure [kPa]"

        elif feature.startswith("total_precipitation"):
            metadata["total_precipitation"] = "Total precipitation [mm]"

        elif feature.startswith("potential_evaporation_sum_ERA5"):
            metadata[
                "potential_evaporation_sum_ERA5_LAND"] = "Potential Evaporation [mm] (original potential_evaporation from ERA5-Land)"

        elif feature.startswith("potential_evaporation_sum_FAO"):
            metadata[
                "potential_evaporation_sum_FAO_PENMAN_MONTEITH"] = "Potential Evaporation [mm] (FAO Penman-Monteith computed from ERA5-Land inputs)"

        elif feature.startswith("u_component_of_wind_10m"):
            metadata["u_component_of_wind_10m"] = "U-component of wind at 10m [m/s]"

        elif feature.startswith("v_component_of_wind_10m"):
            metadata["v_component_of_wind_10m"] = "V-component of wind at 10m [m/s]"

        elif feature.startswith("volumetric_soil_water_layer_1"):
            metadata["volumetric_soil_water_layer_1"] = "ERA5-Land volumetric soil water layer 1 (0-7cm) [m3/m3]"

        elif feature.startswith("volumetric_soil_water_layer_2"):
            metadata["volumetric_soil_water_layer_2"] = "ERA5-Land volumetric soil water layer 2 (7-28cm) [m3/m3]"

        elif feature.startswith("volumetric_soil_water_layer_3"):
            metadata["volumetric_soil_water_layer_3"] = "ERA5-Land volumetric soil water layer 3 (28-100cm) [m3/m3]"

        elif feature.startswith("volumetric_soil_water_layer_4"):
            metadata["volumetric_soil_water_layer_4"] = "ERA5-Land volumetric soil water layer 4 (100-289cm) [m3/m3]"

        elif feature.startswith("streamflow"):
            metadata["streamflow"] = "Observed streamflow [mm/d]"

    return metadata


def _get_offset(tz_name, lat):
    """Convert a time zone to offset from UTC in hours. """
    # Making sure it is always non-DST, depending on northern/southern hemisphere
    if lat <= 0:
        some_date = datetime.strptime("2020-01-01", "%Y-%m-%d")
    else:
        some_date = datetime.strptime("2020-08-01", "%Y-%m-%d")
    tz_target = timezone(tz_name)
    date_lst = tz_target.localize(some_date)
    date_utc = utc.localize(some_date)
    return (date_utc - date_lst).total_seconds() / (60 * 60)


@njit
def _split_list(a_list: List) -> List:
    """Splits list into list of lists, where each list contains subsequent numbers."""
    new_list = []
    start = 0
    for index, value in enumerate(a_list):
        if index < len(a_list) - 1:
            if a_list[index + 1] > value + 1:
                end = index + 1
                new_list.append(a_list[start:end])
                start = end
        else:
            new_list.append(a_list[start:len(a_list)])
    return new_list


def _utc_to_local_standard_time(df: pd.DataFrame, lat, lon) -> pd.DataFrame:
    """Convert a timezone from UTC to local time, given lat/lon."""
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    offset = _get_offset(tz_name, lat)

    df.index = df.index + pd.to_timedelta(offset, unit='h')
    return df
