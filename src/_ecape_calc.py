# Note from Amelia Urquhart:
# The following code is adapted from https://github.com/citylikeamradio/ecape. It is slightly
# modified to allow for custom inflow layers to be used, so the ECAPE calculations won't be
# limited to the 0-1 km RM inflow.

# There are also a few issues related to MetPy weirdness and irreversible adiabatic descent that
# I've worked out here.

# SPDX-FileCopyrightText: 2023-present Robert Capella <bob.capella@gmail.com>
# SPDX-License-Identifier: MIT

"""Calculate the entraining CAPE (ECAPE) of a parcel"""
from typing import Callable, Tuple

import metpy.calc as mpcalc
import numpy as np
import pint
from metpy.constants import dry_air_spec_heat_press, earth_gravity
from metpy.units import check_units, units
import math

PintList = np.typing.NDArray[pint.Quantity]


@check_units("[pressure]", "[temperature]", "[temperature]")
def _get_parcel_profile(
    pressure: PintList, temperature: PintList, dew_point_temperature: PintList, parcel_func: Callable = None
) -> PintList:
    """
    Retrieve a parcel's temperature profile.

    Args:
        pressure:
            Total atmospheric pressure
        temperature:
            Air temperature
        dew_point_temperature:
            Dew point temperature
        parcel_func:
            parcel profile retrieval callable via MetPy

    Returns:
        parcel_profile

    """

    # if surface-based, skip this process, None is default for lfc, el MetPy calcs
    if parcel_func:
        # calculate the parcel's starting temperature, then parcel temperature profile
        parcel_p, parcel_t, parcel_td, *parcel_i = parcel_func(pressure, temperature, dew_point_temperature)
        parcel_profile = mpcalc.parcel_profile(pressure, parcel_t, parcel_td)
    else:
        parcel_profile = None

    return parcel_profile


@check_units("[pressure]", "[length]", "[temperature]", "[temperature]")
def calc_lfc_height(
    pressure: PintList, height_msl: PintList, temperature: PintList, dew_point_temperature: PintList, parcel_func: Callable
) -> Tuple[int, pint.Quantity]:
    """
    Retrieve a parcel's level of free convection (lfc).

    Args:
        pressure:
            Total atmospheric pressure
        height_msl:
            Atmospheric heights at the levels given by 'pressure'.
        temperature:
            Air temperature
        dew_point_temperature:
            Dew point temperature
        parcel_func:
            parcel profile retrieval callable via MetPy

    Returns:
        lfc:
            index of the last instance of negative buoyancy below the lfc
        lfc_z:
            height of the last instance of negative buoyancy below the lfc

    """

    # calculate the parcel's temperature profile
    parcel_profile = _get_parcel_profile(pressure, temperature, dew_point_temperature, parcel_func)

    print("profile:", parcel_func)
    # for i in range(len(temperature)):
    #     print(i, temperature[i], parcel_profile[i].to('degC'))

    # calculate the lfc, select the appropriate index & associated height
    lfc_p, lfc_t = mpcalc.lfc(pressure, temperature, dew_point_temperature, parcel_temperature_profile=parcel_profile)
    
    if(math.isnan(lfc_p.m)):
        return None, None

    lfc_idx = (pressure - lfc_p > 0).nonzero()[0][-1]
    lfc_z = height_msl[lfc_idx]

    return lfc_idx, lfc_z


@check_units("[pressure]", "[length]", "[temperature]", "[temperature]")
def calc_el_height(
    pressure: PintList, height_msl: PintList, temperature: PintList, dew_point_temperature: PintList, parcel_func: Callable
) -> Tuple[int, pint.Quantity]:
    """
    Retrieve a parcel's equilibrium level (el).

    Args:
        pressure:
            Total atmospheric pressure
        height_msl:
            Atmospheric heights at the levels given by 'pressure'.
        temperature:
            Air temperature
        dew_point_temperature:
            Dew point temperature
        parcel_func:
            parcel profile retrieval callable via MetPy

    Returns:
        el_idx:
            index of the last instance of positive buoyancy below the el
        el_z:
            height of the last instance of positive buoyancy below the el

    """

    # calculate the parcel's temperature profile
    parcel_profile = _get_parcel_profile(pressure, temperature, dew_point_temperature, parcel_func)

    # calculate the el, select the appropriate index & associated height
    el_p, el_t = mpcalc.el(pressure, temperature, dew_point_temperature, parcel_temperature_profile=parcel_profile)
    
    if(math.isnan(el_p.m)):
        return None, None
    
    el_idx = (pressure - el_p > 0).nonzero()[0][-1]
    el_z = height_msl[el_idx]

    return el_idx, el_z


@check_units("[pressure]", "[speed]", "[speed]", "[length]")
def calc_sr_wind(pressure: PintList, u_wind: PintList, v_wind: PintList, height_msl: PintList, infl_bottom: pint.Quantity = 0 * units("m"), infl_top: pint.Quantity = 1000 * units("m"), storm_motion_type: str = "right_moving") -> pint.Quantity:
    """
    Calculate the mean storm relative (as compared to Bunkers right motion) wind magnitude in the 0-1 km AGL layer

    Modified by Amelia Urquhart to allow for custom inflow layers as well as a choice between Bunkers right, Bunkers left, and Mean Wind

    Args:
        pressure:
            Total atmospheric pressure
        u_wind:
            X component of the wind
        v_wind
            Y component of the wind
        height_msl:
            Atmospheric heights at the levels given by 'pressure'.

    Returns:
        sr_wind:
            0-1 km AGL average storm relative wind magnitude

    """
    height_agl = height_msl - height_msl[0]
    bunkers_right, bunkers_left, mean_wind = mpcalc.bunkers_storm_motion(pressure, u_wind, v_wind, height_agl)  # right, left, mean

    u_sr = None
    v_sr = None
    
    if "left_moving" == storm_motion_type:
        u_sr = u_wind - bunkers_left[0]  # u-component
        v_sr = v_wind - bunkers_left[1]  # v-component
    elif "mean_wind" == storm_motion_type:
        u_sr = u_wind - mean_wind[0]  # u-component
        v_sr = v_wind - mean_wind[1]  # v-component
    else:
        u_sr = u_wind - bunkers_right[0]  # u-component
        v_sr = v_wind - bunkers_right[1]  # v-component

    u_sr_1km = u_sr[np.nonzero((height_agl >= infl_bottom) & (height_agl <= infl_top))]
    v_sr_1km = v_sr[np.nonzero((height_agl >= infl_bottom) & (height_agl <= infl_top))]

    sr_wind = np.mean(mpcalc.wind_speed(u_sr_1km, v_sr_1km))

    return sr_wind


@check_units("[pressure]", "[length]", "[temperature]", "[mass]/[mass]")
def calc_mse(
    pressure: PintList, height_msl: PintList, temperature: PintList, specific_humidity: PintList
) -> Tuple[PintList, PintList]:
    """
    Calculate the moist static energy terms of interest.

    Args:
        pressure:
            Total atmospheric pressure
        height_msl:
            Atmospheric heights at the levels given by 'pressure'.
        temperature:
            Air temperature
        specific_humidity:
            Specific humidity

    Returns:
        moist_static_energy_bar:
            Mean moist static energy from the surface to a layer
        moist_static_energy_star:
            Saturated moist static energy
    """

    # calculate MSE_bar
    moist_static_energy = mpcalc.moist_static_energy(height_msl, temperature, specific_humidity)
    moist_static_energy_bar = np.cumsum(moist_static_energy) / np.arange(1, len(moist_static_energy) + 1)
    moist_static_energy_bar = moist_static_energy_bar.to("J/kg")

    # calculate MSE*
    saturation_mixing_ratio = mpcalc.saturation_mixing_ratio(pressure, temperature)
    moist_static_energy_star = mpcalc.moist_static_energy(height_msl, temperature, saturation_mixing_ratio)
    moist_static_energy_star = moist_static_energy_star.to("J/kg")

    return moist_static_energy_bar, moist_static_energy_star


@check_units("[energy]/[mass]", "[energy]/[mass]", "[temperature]")
def calc_integral_arg(moist_static_energy_bar, moist_static_energy_star, temperature) -> PintList:
    """
    Calculate the contents of the integral defined in the NCAPE equation (54).

    Args:
        moist_static_energy_bar:
            Mean moist static energy from the surface to a layer
        moist_static_energy_star:
            Saturated moist static energy
        temperature:
            Air temperature

    Returns:
        integral_arg:
            Contents of integral defined in NCAPE eqn. 54

    """

    # NCAPE eqn 54 integrand, see compute_NCAPE.m L32
    integral_arg = -(earth_gravity / (dry_air_spec_heat_press * temperature)) * (
        moist_static_energy_bar - moist_static_energy_star
    )

    return integral_arg


@check_units("[length]/[time]**2", "[length]", "[dimensionless]", "[dimensionless]")
def calc_ncape(integral_arg: PintList, height_msl: PintList, lfc_idx: int, el_idx: int) -> pint.Quantity:
    """
    Calculate the buoyancy dilution potential (NCAPE)

    Args:
        integral_arg:
            Contents of integral defined in NCAPE eqn. 54
        height_msl:
            Atmospheric heights at the levels given by 'pressure'.
        lfc_idx:
            Index of the last instance of negative buoyancy below the lfc
        el_idx:
            Index of the last instance of positive buoyancy below the el

    Returns:
        ncape:
            Buoyancy dilution potential of the free troposphere (eqn. 54)
    """

    # see compute_NCAPE.m L41
    ncape = np.sum(
        (0.5 * integral_arg[lfc_idx:el_idx] + 0.5 * integral_arg[lfc_idx + 1 : el_idx + 1])
        * (height_msl[lfc_idx + 1 : el_idx + 1] - height_msl[lfc_idx:el_idx])
    )

    return ncape


@check_units("[speed]", "[dimensionless]", "[length]**2/[time]**2", "[energy]/[mass]")
def calc_ecape_a(sr_wind: PintList, psi: pint.Quantity, ncape: pint.Quantity, cape: pint.Quantity) -> pint.Quantity:
    """
    Calculate the entraining cape of a parcel

    Args:
        sr_wind:
            0-1 km AGL average storm relative wind magnitude
        psi:
            Parameter defined in eqn. 52, constant for a given equilibrium level
        ncape:
            Buoyancy dilution potential of the free troposphere (eqn. 54)
        cape:
            Convective available potential energy (CAPE, user-defined type)
    Returns:
        ecape:
            Entraining CAPE (eqn. 55)
    """

    # broken into terms for readability
    term_a = sr_wind**2 / 2.0
    term_b = (-1 - psi - (2 * psi / sr_wind**2) * ncape) / (4 * psi / sr_wind**2)
    term_c = (
        np.sqrt((1 + psi + (2 * psi / sr_wind**2) * ncape) ** 2 + 8 * (psi / sr_wind**2) * (cape - (psi * ncape)))
    ) / (4 * psi / sr_wind**2)

    ecape_a = term_a + term_b + term_c

    # set to 0 if negative
    return ecape_a.to("J/kg") if ecape_a >= 0 else 0


@check_units("[length]")
def calc_psi(el_z: pint.Quantity) -> pint.Quantity:
    """
    Calculate the constant psi as denoted in eqn. 52

    Args:
        el_z:
            height of the last instance of positive buoyancy below the el

    Returns:
        psi:
            Parameter defined in eqn. 52, constant for a given equilibrium level, see COMPUTE_ECAPE.m L88 (pitchfork)
    """

    # additional constants as denoted in section 4 step 1.
    sigma = 1.6 * units("dimensionless")
    alpha = 0.8 * units("dimensionless")
    l_mix = 120.0 * units("m")
    pr = (1.0 / 3.0) * units("dimensionless")  # prandtl number
    ksq = 0.18 * units("dimensionless")  # von karman constant

    psi = (ksq * alpha**2 * np.pi**2 * l_mix) / (4.0 * pr * sigma**2 * el_z)

    return psi


@check_units("[length]", "[pressure]", "[temperature]", "[mass]/[mass]", "[speed]", "[speed]")
def calc_ecape(
    height_msl: PintList,
    pressure: PintList,
    temperature: PintList,
    specific_humidity: PintList,
    u_wind: PintList,
    v_wind: PintList,
    cape_type: str = "most_unstable",
    undiluted_cape: pint.Quantity = None,
    inflow_bottom: pint.Quantity = 0 * units("m"), 
    inflow_top: pint.Quantity = 1000 * units("m"), 
    storm_motion: str = "right_moving",
    lfc: pint.Quantity = None, 
    el: pint.Quantity = None, 
) -> pint.Quantity:
    """
    Calculate the entraining CAPE (ECAPE) of a parcel

    Parameters:
    ------------
        height_msl: np.ndarray[pint.Quantity]
            Atmospheric heights at the levels given by 'pressure' (MSL)
        pressure: np.ndarray[pint.Quantity]
            Total atmospheric pressure
        temperature: np.ndarray[pint.Quantity]
            Air temperature
        specific humidity: np.ndarray[pint.Quantity]
            Specific humidity
        u_wind: np.ndarray[pint.Quantity]
            X component of the wind
        v_wind np.ndarray[pint.Quantity]
            Y component of the wind
        cape_type: str
            Variation of CAPE desired. 'most_unstable' (default), 'surface_based', or 'mixed_layer'
        undiluted_cape: pint.Quantity
            User-provided undiluted CAPE value

    Returns:
    ----------
        ecape : 'pint.Quantity'
            Entraining CAPE
    """

    cape_func = {
        "most_unstable": mpcalc.most_unstable_cape_cin,
        "surface_based": mpcalc.surface_based_cape_cin,
        "mixed_layer": mpcalc.mixed_layer_cape_cin,
    }

    parcel_func = {
        "most_unstable": mpcalc.most_unstable_parcel,
        "surface_based": None,
        "mixed_layer": mpcalc.mixed_parcel,
    }

    # calculate cape
    dew_point_temperature = mpcalc.dewpoint_from_specific_humidity(pressure, temperature, specific_humidity)

    # whether the user has not / has overidden the cape calculations
    if not undiluted_cape:
        cape, _ = cape_func[cape_type](pressure, temperature, dew_point_temperature)
    else:
        cape = undiluted_cape

    lfc_idx = None
    el_idx = None
    el_z = None

    # print("cape_type:", cape_type)
    # print("parcel_func:", parcel_func[cape_type])

    if lfc == None:
        # print("doing lfc_idx as calc lfc height")
        # calculate the level of free convection (lfc) and equilibrium level (el) indexes
        lfc_idx, _ = calc_lfc_height(pressure, height_msl, temperature, dew_point_temperature, parcel_func[cape_type])
        el_idx, el_z = calc_el_height(pressure, height_msl, temperature, dew_point_temperature, parcel_func[cape_type])
    else:
        # print("doing lfc_idx as np where")
        lfc_idx = np.where(height_msl > lfc)[0][0]
        el_idx = np.where(height_msl > el)[0][0]
    #     print(i, temperature[i], parcel_profile[i].to('degC'))
        el_z = el

    # calculate the buoyancy dilution potential (ncape)
    moist_static_energy_bar, moist_static_energy_star = calc_mse(pressure, height_msl, temperature, specific_humidity)
    integral_arg = calc_integral_arg(moist_static_energy_bar, moist_static_energy_star, temperature)
    ncape = calc_ncape(integral_arg, height_msl, lfc_idx, el_idx)

    # calculate the storm relative (sr) wind
    sr_wind = calc_sr_wind(pressure, u_wind, v_wind, height_msl, infl_bottom=inflow_bottom, infl_top=inflow_top, storm_motion_type=storm_motion)

    # calculate the entraining cape (ecape)
    psi = calc_psi(el_z)
    ecape_a = calc_ecape_a(sr_wind, psi, ncape, cape)

    return ecape_a


if __name__ == "__main__":
    pass
