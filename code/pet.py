"""Functions to calculate PET following FAO Penman-Monteith.

The code from this module is largely copied from 
https://github.com/Dagmawi-TA/hPET/blob/main/pet_calc_v3.3.py, which is the published code for 
Singer et al. (2021), see https://www.nature.com/articles/s41597-021-01003-9. Only minor 
modifications were made to make the functions fit into the Caravan code base.
"""

import numpy as np
import pandas as pd


def get_fao_pm_pet(
    surface_pressure_mean: pd.Series,
    temperature_2m_mean: pd.Series,
    dewpoint_temperature_2m_mean: pd.Series,
    u_component_of_wind_10m_mean: pd.Series,
    v_component_of_wind_10m_mean: pd.Series,
    surface_net_solar_radiation_mean: pd.Series,
    surface_net_thermal_radiation_mean: pd.Series,
) -> pd.Series:
    """Returns Penman-Monteith PET as a pandas Series.

    All inputs to these functions should be from the dataframe that is returned by 
    caravan_utils.aggregate_df_to_daily() and hence after the Caravan specific unit conversion.
    
    Parameters
    ----------
    surface_pressure_mean : pd.Series
        Daily mean surface pressure.
    temperature_2m_mean: pd.Series
        Daily mean temperature.
    dewpoint_temperature_2m_mean: pd.Series
        Daily mean dewpoint temperature.
    u_component_of_wind_10m_mean: pd.Series
        Daily mean u-component of wind.
    v_component_of_wind_10m_mean: pd.Series
        Daily mean v-component of wind.
    surface_net_solar_radiation_mean: pd.Series
        Mean net solar radiation.
    surface_net_thermal_radiation_mean: pd.Series
        Mean net thermal radiation.

    Returns
    -------
    Pandas Series with Penman-Monteith PET following the FAO guidelines.
    """
    windspeed2m_m_s, net_radiation_MJ_m2 = _preprocess_inputs(
        u_component_of_wind_10m_mean=u_component_of_wind_10m_mean,
        v_component_of_wind_10m_mean=v_component_of_wind_10m_mean,
        surface_net_solar_radiation_mean=surface_net_solar_radiation_mean,
        surface_net_thermal_radiation_mean=surface_net_thermal_radiation_mean)

    pm_pet = _calculate_pm_pet_daily(
        surface_pressure_kpa=surface_pressure_mean,
        temperature2m_c=temperature_2m_mean,
        dewpoint2m_c=dewpoint_temperature_2m_mean,
        windspeed2m_m_s=windspeed2m_m_s,
        net_radiation_mj_m2=net_radiation_MJ_m2,
    )

    return pm_pet.clip(lower=0.0)  # Clip negative PET values to zero.


def _preprocess_inputs(
    u_component_of_wind_10m_mean: pd.Series,
    v_component_of_wind_10m_mean: pd.Series,
    surface_net_solar_radiation_mean: pd.Series,
    surface_net_thermal_radiation_mean: pd.Series,
) -> pd.Series:
    temp_windspeed10m_m_s = np.sqrt(u_component_of_wind_10m_mean**2 + v_component_of_wind_10m_mean**2)
    windspeed2m_m_s = temp_windspeed10m_m_s * 4.87 / (np.log(67.8 * 10 - 5.42))

    net_radiation_MJ_m2 = ((surface_net_solar_radiation_mean + surface_net_thermal_radiation_mean) * 3600 * 24 / 1e6)
    return windspeed2m_m_s, net_radiation_MJ_m2


def _calculate_pm_pet_daily(
        surface_pressure_kpa: pd.Series,  # surface pressure KPa
        temperature2m_c: pd.Series,  # Daily mean temperature at 2 m
        dewpoint2m_c: pd.Series,  # Daily mean dewpoint temperature at 2 m
        windspeed2m_m_s: pd.Series,  # Windspeed at 2 m
        net_radiation_mj_m2: pd.Series,  # Total daily net downward radiation MJ/m2/day
) -> np.ndarray:
    # Constants.
    lmbda = 2.45  # Latent heat of vaporization [MJ kg -1] (simplification in the FAO PenMon (latent heat of about 20°C)
    cp = 1.013e-3  # Specific heat at constant pressure [MJ kg-1 °C-1]
    eps = 0.622  # Ratio molecular weight of water vapour/dry air

    # Soil heat flux density [MJ m-2 day-1] - set to 0 following eq 42 in FAO
    soil_heat_flux = np.zeros_like(surface_pressure_kpa)

    # Atmospheric pressure [kPa] eq 7 in FAO.
    P_kpa = surface_pressure_kpa

    # Psychrometric constant (gamma symbol in FAO) eq 8 in FAO.
    psychometric_kpa_c = cp * P_kpa / (eps * lmbda)

    # Saturation vapour pressure, eq 11 in FAO.
    svp_kpa = 0.6108 * np.exp((17.27 * temperature2m_c) / (temperature2m_c + 237.3))

    # Delta (slope of saturation vapour pressure curve) eq 13 in FAO.
    delta_kpa_c = 4098.0 * svp_kpa / (temperature2m_c + 237.3)**2

    # Actual vapour pressure, eq 14 in FAO.
    avp_kpa = 0.6108 * np.exp((17.27 * dewpoint2m_c) / (dewpoint2m_c + 237.3))

    # Saturation vapour pressure deficit.
    svpdeficit_kpa = svp_kpa - avp_kpa

    # Calculate ET0, equation 6 in FAO
    numerator = (0.408 * delta_kpa_c * (net_radiation_mj_m2 - soil_heat_flux) + psychometric_kpa_c *
                 (900 / (temperature2m_c + 273)) * windspeed2m_m_s * svpdeficit_kpa)
    denominator = delta_kpa_c + psychometric_kpa_c * (1 + 0.34 * windspeed2m_m_s)

    ET0_mm_day = numerator / denominator
    return ET0_mm_day
