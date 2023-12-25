import metpy as mpy
import metpy.calc as mpcalc
import pint
import math
import numpy as np
import sys

from ecape.calc import calc_ecape, calc_el_height
from metpy.units import check_units, units
from pint import UnitRegistry
ureg = UnitRegistry()

PintList = np.typing.NDArray[pint.Quantity]

# relevant ECAPE constants
sigma = 1.6
alpha = 0.8
k2 = 0.18
L_mix = 120
Pr = 1/3

# @param cape                   Units: J kg^-1
# @param ecape                  Units: J kg^-1
# @param vsr                    Units: m s^-1
# @param storm_column_height    Units: Meters
# @return updraft_radius:       Units: Meters
def updraft_radius(cape: float, ecape: float, vsr: float, storm_column_height: float) -> float:
    nondim_e = ecape / cape
    nondim_v = vsr / math.sqrt(2 * ecape)

    nondim_r = math.sqrt(((4 * sigma * sigma) / (alpha * alpha * math.pi * math.pi)) * ((nondim_v * nondim_v)/(nondim_e)))

    updraft_radius = nondim_r * storm_column_height

    return updraft_radius/2

# @param updraftRadius              Units: Meters
# @return entrainment_rate:         Units: m^-1
def entrainment_rate(updraft_radius: float) -> float:
    entrainment_rate = (2 * k2 * L_mix) / (Pr * updraft_radius * updraft_radius)

    return entrainment_rate

# LOOK THROUGH KYLE GILLETT'S CODE TO SEE HOW HE DECIDED TO DO THESE PROFILES
# CITYLIKEAMRADIO'S TOO

# Unlike the Java version, this expects arrays sorted in order of increasing height, decreasing pressure
# This is to keep in line with MetPy conventions
@check_units("[pressure]", "[length]", "[temperature]", "[temperature]", "[speed]", "[speed]")
def ecape_parcel(
        pressure: PintList, 
        height: PintList, 
        temperature: PintList, 
        dewpoint: PintList, 
        u_wind: PintList, 
        v_wind: PintList,
        cape_type: str = "most_unstable",
        storm_motion_type: str = "right_moving", 
        inflow_bottom: pint.Quantity = 0 * ureg.kilometer, 
        inflow_top: pint.Quantity = 1 * ureg.kilometer, 
        cape: pint.Quantity = None,
        el: pint.Quantity = None) -> None: # trying to return whatever metpy returns but i'm having trouble finding out what that is. they don't type annotate their code
    if cape_type not in ['most_unstable', 'mixed_layer', 'surface_based']:
        sys.exit("Invalid 'method' kwarg. Valid methods inlcude ['era5', 'rap-ruc', 'rap-now', 'ncep-fnl']")

    specific_humidity = mpcalc.specific_humidity_from_dewpoint(pressure, dewpoint)

    parcel_pressure = -1024
    parcel_height = -1024
    parcel_temperature = -1024
    parcel_dewpoint = -1024

    parcel_func = {
        "most_unstable": mpcalc.most_unstable_parcel,
        "surface_based": None,
        "mixed_layer": mpcalc.mixed_parcel,
    }

    if "most_unstable" == cape_type:
        parcel_pressure, parcel_temperature, parcel_dewpoint, mu_idx = mpcalc.most_unstable_parcel(pressure, temperature, dewpoint)
        parcel_height = height[mu_idx]
    elif "mixed_layer" == cape_type:
        parcel_temperature, parcel_dewpoint = mpcalc.mixed_layer(pressure, temperature, dewpoint)
        parcel_pressure, _, _ = mpcalc.mixed_parcel(pressure, temperature, dewpoint)
        parcel_height = height[0]
    elif "surface_based" == cape_type:
        parcel_pressure = pressure[0]
        parcel_height = height[0]
        parcel_temperature = temperature[0]
        parcel_dewpoint[0]
        
    if cape == None and el == None:
        parcel_profile = mpcalc.parcel_profile(pressure, parcel_temperature, parcel_dewpoint)
        cape, _ = mpcalc.cape_cin(pressure, temperature, dewpoint, parcel_profile)
        el = calc_el_height(pressure, height, temperature, dewpoint, parcel_func[cape_type])[1]
    elif cape != None and el == None:
        parcel_profile = mpcalc.parcel_profile(pressure, parcel_temperature, parcel_dewpoint)
        el = calc_el_height(pressure, temperature, dewpoint, parcel_profile)
    elif cape == None and el != None:
        parcel_profile = mpcalc.parcel_profile(pressure, parcel_temperature, parcel_dewpoint)
        cape, _ = mpcalc.cape_cin(pressure, temperature, dewpoint, parcel_profile) * units("J/kg")

    # print("el: ", el)
    # print("parcel_height: ", parcel_height)

    ecape = calc_ecape(height, pressure, temperature, specific_humidity, u_wind, v_wind, cape_type, cape)
    vsr = calc_sr_wind(pressure, u_wind, v_wind, height, storm_motion_type, inflow_bottom, inflow_top)
    storm_column_height = el - parcel_height

    cape = cape.to("joule / kilogram")
    ecape = ecape.to("joule / kilogram")
    vsr = vsr.to("meter / second")
    storm_column_height = storm_column_height.to("meter")

    print("cape: ", cape)
    print("ecape: ", ecape)
    print("vsr: ", vsr)
    print("storm_column_height: ", storm_column_height)

    r = updraft_radius(cape.magnitude, ecape.magnitude, vsr.magnitude, storm_column_height.magnitude)
    epsilon = entrainment_rate(r)

    print("updr: ", r, " m")
    print("entr: ", epsilon, " m^-1")

    return None

# borrowed from github.com/citylikeamradio/ecape and adjusted
@check_units("[pressure]", "[speed]", "[speed]", "[length]")
def calc_sr_wind(pressure: PintList, u_wind: PintList, v_wind: PintList, height_msl: PintList, storm_motion_type: str = "right_moving", inflow_bottom: pint.Quantity = 0 * units("m"), inflow_top: pint.Quantity = 1000 * units("m")) -> pint.Quantity:
    """
    Calculate the mean storm relative (as compared to Bunkers right motion) wind magnitude in the 0-1 km AGL layer

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
    bunkers_right, bunkers_left, bunkers_mean = mpcalc.bunkers_storm_motion(pressure, u_wind, v_wind, height_agl)  # right, left, mean

    storm_motion = None

    if("right_moving" == storm_motion_type):
        storm_motion = bunkers_right
    elif("left_moving" == storm_motion_type):
        storm_motion = bunkers_left
    elif("mean_wind" == storm_motion_type):
        storm_motion = bunkers_mean

    u_sr = u_wind - storm_motion[0]  # u-component
    v_sr = v_wind - storm_motion[1]  # v-component

    u_sr_ = u_sr[np.nonzero((height_agl >= inflow_bottom.magnitude * units("m")) & (height_agl <= inflow_top.magnitude * units("m")))]
    v_sr_ = v_sr[np.nonzero((height_agl >= inflow_bottom.magnitude * units("m")) & (height_agl <= inflow_top.magnitude * units("m")))]

    sr_wind = np.mean(mpcalc.wind_speed(u_sr_, v_sr_))

    return sr_wind

#updr = updraft_radius(4539, 2993, 10, 12000)
#print("updr: ", updr, " m")
#entr = entrainment_rate(updr)
#print("entr: ", entr, " m^-1")