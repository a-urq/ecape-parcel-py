#
# AUTHOR: Amelia Urquhart (https://github.com/a-urq)
# VERSION: 1.0.5.1
# DATE: December 25, 2023
#

import metpy as mpy
import metpy.calc as mpcalc
import pint
import math
import numpy as np
import sys

from ecape.calc import calc_ecape, calc_el_height
from metpy.units import check_units, units
from pint import UnitRegistry

from typing import Tuple

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

ECAPE_PARCEL_DZ: pint.Quantity = 20 * units.meter
DRY_ADIABATIC_LAPSE_RATE: pint.Quantity = 9.8 * units.kelvin / units.kilometer
DEWPOINT_LAPSE_RATE: pint.Quantity = 1.8 * units.kelvin / units.kilometer

# Unlike the Java version, this expects arrays sorted in order of increasing height, decreasing pressure
# This is to keep in line with MetPy conventions
# Returns Tuple of { parcel_pressure, parcel_height, parcel_temperature, parcel_dewpoint }
@check_units("[pressure]", "[length]", "[temperature]", "[temperature]", "[speed]", "[speed]")
def calc_ecape_parcel(
        pressure: PintList, 
        height: PintList, 
        temperature: PintList, 
        dewpoint: PintList, 
        u_wind: PintList, 
        v_wind: PintList,
        align_to_input_pressure_values: bool,
        cape_type: str = "most_unstable",
        storm_motion_type: str = "right_moving", 
        inflow_bottom: pint.Quantity = 0 * units.kilometer, 
        inflow_top: pint.Quantity = 1 * units.kilometer, 
        cape: pint.Quantity = None,
        el: pint.Quantity = None) -> Tuple[pint.Quantity, pint.Quantity, pint.Quantity, pint.Quantity]:
    
    if cape_type not in ['most_unstable', 'mixed_layer', 'surface_based']:
        sys.exit("Invalid 'cape_type' kwarg. Valid cape_types include ['most_unstable', 'mixed_layer', 'surface_based']")
    
    if storm_motion_type not in ['right_moving', 'left_moving', 'mean_wind']:
        sys.exit("Invalid 'storm_motion_type' kwarg. Valid storm_motion_types include ['right_moving', 'left_moving', 'mean_wind']")

    specific_humidity = mpcalc.specific_humidity_from_dewpoint(pressure, dewpoint)
    moist_static_energy = mpcalc.moist_static_energy(height, temperature, specific_humidity).to("J/kg")

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
        parcel_dewpoint = dewpoint[0]
        
    if cape == None and el == None:
        parcel_profile = mpcalc.parcel_profile(pressure, parcel_temperature, parcel_dewpoint)
        cape, _ = mpcalc.cape_cin(pressure, temperature, dewpoint, parcel_profile)
        el = calc_el_height(pressure, height, temperature, dewpoint, parcel_func[cape_type])[1]
    elif cape != None and el == None:
        parcel_profile = mpcalc.parcel_profile(pressure, parcel_temperature, parcel_dewpoint)
        el = calc_el_height(pressure, height, temperature, dewpoint, parcel_func[cape_type])[1]
    elif cape == None and el != None:
        parcel_profile = mpcalc.parcel_profile(pressure, parcel_temperature, parcel_dewpoint)
        cape, _ = mpcalc.cape_cin(pressure, temperature, dewpoint, parcel_profile) * units("J/kg")

    # print("el: ", el)
    # print("parcel_height: ", parcel_height)
        
    # print(height)
    # print(pressure)
    # print(temperature)
    # print(specific_humidity)
    # print(u_wind)
    # print(v_wind)
    # print(cape_type)
    # print(cape)
        
    if cape <= 0:
        if align_to_input_pressure_values:
            pressure_raw = [None] * len(pressure)
            height_raw = [None] * len(pressure)
            temperature_raw = [None] * len(pressure)
            dewpoint_raw = [None] * len(pressure)

            return (pressure_raw,  height_raw, temperature_raw, dewpoint_raw)
        else:
            pressure_raw = [None]
            height_raw = [None]
            temperature_raw = [None]
            dewpoint_raw = [None]
            
            return (pressure_raw,  height_raw, temperature_raw, dewpoint_raw)


    ecape = calc_ecape(height, pressure, temperature, specific_humidity, u_wind, v_wind, cape_type, cape)
    vsr = calc_sr_wind(pressure, u_wind, v_wind, height, storm_motion_type, inflow_bottom, inflow_top)
    storm_column_height = el - parcel_height

    cape = cape.to("joule / kilogram")
    ecape = ecape.to("joule / kilogram")
    vsr = vsr.to("meter / second")
    storm_column_height = storm_column_height.to("meter")

    #print("cape: ", cape)
    #print("ecape: ", ecape)
    #print("vsr: ", vsr)
    #print("storm_column_height: ", storm_column_height)

    r = updraft_radius(cape.magnitude, ecape.magnitude, vsr.magnitude, storm_column_height.magnitude)
    epsilon = entrainment_rate(r)

    #print("updr: ", r, " m")
    #print("entr: ", epsilon, " m^-1")

    parcel_temperature = parcel_temperature.to("degK")
    parcel_dewpoint = parcel_dewpoint.to("degK")

    pressure_raw = []
    height_raw = []
    temperature_raw = []
    dewpoint_raw = []

    pressure_raw.append(parcel_pressure)
    height_raw.append(parcel_height)
    temperature_raw.append(parcel_temperature)
    dewpoint_raw.append(parcel_dewpoint)

    parcel_specific_humidity = mpcalc.specific_humidity_from_dewpoint(parcel_pressure, parcel_dewpoint)
    parcel_moist_static_energy = mpcalc.moist_static_energy(parcel_height, parcel_temperature, parcel_specific_humidity)

    while parcel_pressure >= pressure[-1]:
        parcel_pressure = pressure_at_height(parcel_pressure, ECAPE_PARCEL_DZ, parcel_temperature)
        parcel_height += ECAPE_PARCEL_DZ

        if parcel_dewpoint < parcel_temperature:
            parcel_temperature -= DRY_ADIABATIC_LAPSE_RATE * ECAPE_PARCEL_DZ
            parcel_dewpoint -= DEWPOINT_LAPSE_RATE * ECAPE_PARCEL_DZ

            #parcel_specific_humidity = mpcalc.specific_humidity_from_dewpoint(parcel_pressure, parcel_dewpoint)
            #env_specific_humidity = linear_interp(height, specific_humidity, parcel_height)

            #dq = -epsilon * (parcel_specific_humidity - env_specific_humidity) / units.meter

            #parcel_specific_humidity += dq * ECAPE_PARCEL_DZ
            
            parcel_dewpoint = mpcalc.dewpoint_from_specific_humidity(parcel_pressure, parcel_temperature, parcel_specific_humidity).to("degK")

            parcel_moist_static_energy = mpcalc.moist_static_energy(parcel_height, parcel_temperature, parcel_specific_humidity)
        else:
            env_moist_static_energy = linear_interp(height, moist_static_energy, parcel_height)

            dh = -epsilon * (parcel_moist_static_energy - env_moist_static_energy) / units.meter

            parcel_moist_static_energy += dh * ECAPE_PARCEL_DZ

            parcel_temperature_from_mse = temperature_from_mse(parcel_moist_static_energy, parcel_pressure, parcel_height) # this is the weirdest part of the whole package trust me

            parcel_temperature = parcel_temperature_from_mse
            parcel_dewpoint = parcel_temperature_from_mse

        pressure_raw.append(parcel_pressure)
        height_raw.append(parcel_height)
        temperature_raw.append(parcel_temperature)
        dewpoint_raw.append(parcel_dewpoint)

    pressure_units = pressure_raw[-1].units
    height_units = height_raw[-1].units
    temperature_units = temperature_raw[-1].units
    dewpoint_units = dewpoint_raw[-1].units

    #print(pressure_units)
    #print(height_units)
    #print(temperature_units)
    #print(dewpoint_units)

    #print(pressure_raw[0:3])
    #print(height_raw[0:3])
    # print("input_height", height[0:30])
    # print("height_raw", height_raw[0:30])
    # print("temperature_raw", temperature_raw[0:30])
    # print("dewpoint_raw", dewpoint_raw[0:30])

    # for i in range(len(height_raw)):
    #     print(pressure_raw[i], height_raw[i], temperature_raw[i], dewpoint_raw[i])

    pressure_nondim = [pressure.magnitude for pressure in pressure_raw]
    height_nondim = [height.magnitude for height in height_raw]
    temperature_nondim = [temperature.magnitude for temperature in temperature_raw]
    dewpoint_nondim = [dewpoint.magnitude for dewpoint in dewpoint_raw]

    # makes it work ok with sounderpy
    if align_to_input_pressure_values:
        pressure_nondim_aligned = []
        height_nondim_aligned = []
        temperature_nondim_aligned = []
        dewpoint_nondim_aligned = []

        #print(temperature_nondim)
        #print(dewpoint_nondim)
        #print(linear_interp(height, temperature_nondim, 1000 * units.meter))
        #print("above is debugging lin interp")

        for i in range(len(height)):
            input_pressure = pressure[i]
            input_height = height[i]

            # print("searching for interp_t at height", input_height)

            new_t = rev_linear_interp(pressure_raw, temperature_nondim, input_pressure)
            new_td = rev_linear_interp(pressure_raw, dewpoint_nondim, input_pressure)

            pressure_nondim_aligned.append(input_pressure.magnitude)
            height_nondim_aligned.append(input_height.magnitude)
            temperature_nondim_aligned.append(new_t)
            dewpoint_nondim_aligned.append(new_td)

        pressure_nondim = pressure_nondim_aligned
        height_nondim = height_nondim_aligned
        temperature_nondim = temperature_nondim_aligned
        dewpoint_nondim = dewpoint_nondim_aligned

    pressure_qty : pint.Quantity = pressure_nondim * pressure_units
    height_qty : pint.Quantity = height_nondim * height_units
    temperature_qty : pint.Quantity = temperature_nondim * temperature_units
    dewpoint_qty : pint.Quantity = dewpoint_nondim * dewpoint_units

    return ( pressure_qty, height_qty, temperature_qty, dewpoint_qty )

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

molar_gas_constant = 8.314 * units.joule / units.kelvin / units.mole
avg_molar_mass = 0.029 * units.kilogram / units.mole
g = 9.81 * units.meter / units.second / units.second

def pressure_at_height(ref_pressure: pint.Quantity, height_above_ref_pressure: pint.Quantity, temperature: pint.Quantity) -> pint.Quantity:
    temperature = temperature.to("degK")
    height_above_ref_pressure = height_above_ref_pressure.to("m")

    scale_height = (molar_gas_constant * temperature) / (avg_molar_mass * g)

    scale_height = scale_height.magnitude * units.meter

    # print(ref_pressure)
    # print(height_above_ref_pressure)
    # print(scale_height)
    # print(-height_above_ref_pressure.magnitude / scale_height.magnitude)
    # print(math.exp(-height_above_ref_pressure.magnitude / scale_height.magnitude))

    return ref_pressure * math.exp(-height_above_ref_pressure.magnitude / scale_height.magnitude)

def linear_interp(input_arr: PintList, output_arr: PintList, input: pint.Quantity, debug: bool = False) -> pint.Quantity:
    if input < input_arr[0]:
        if debug: print("tree 1")
        return output_arr[0]
    elif input >= input_arr[-1]:
        if debug: print("tree 2")
        return output_arr[-1]
    else:
        for i in range(len(input_arr) - 1):
            input_1 = input_arr[i]
            input_2 = input_arr[i + 1]

            if input == input_1:
                if debug: 
                    print("tree 3 - 1")
                    print(input)
                    print(input_1)

                return output_arr[i]
            elif input < input_2:
                if debug: print("tree 3 - 2")
                output_1 = output_arr[i]
                output_2 = output_arr[i + 1]

                weight_1 = (input_2 - input) / (input_2 - input_1)
                weight_2 = (input - input_1) / (input_2 - input_1)

                return output_1 * weight_1 + output_2 * weight_2
            else:
                continue

    return None # case should not be reached

def rev_linear_interp(input_arr: PintList, output_arr: PintList, input: pint.Quantity, debug: bool = False) -> pint.Quantity:
    if input > input_arr[0]:
        if debug: print("tree 1")
        return output_arr[0]
    elif input <= input_arr[-1]:
        if debug: print("tree 2")
        return output_arr[-1]
    else:
        for i in range(len(input_arr) - 1):
            input_1 = input_arr[i]
            input_2 = input_arr[i + 1]

            if input == input_1:
                if debug: 
                    print("tree 3 - 1")
                    print(input)
                    print(input_1)

                return output_arr[i]
            elif input > input_2:
                if debug: print("tree 3 - 2")
                output_1 = output_arr[i]
                output_2 = output_arr[i + 1]

                weight_1 = (input_2 - input) / (input_2 - input_1)
                weight_2 = (input - input_1) / (input_2 - input_1)

                return output_1 * weight_1 + output_2 * weight_2
            else:
                continue

    return None # case should not be reached

c_p = 1005 * units.joule / units.kilogram / units.kelvin
# iterative solver for the temperature and dewpoint of the parcel from MSE, assuming 100% saturation.
# messy and stupid, but necessary
def temperature_from_mse(mse: pint.Quantity, pressure: pint.Quantity, height: pint.Quantity) -> pint.Quantity:
    #print(mse)
    mse = mse.to("J/kg")
    #print(mse)

    moist_nonstatic_energy = mse - height * g # made up parameter lol

    #print(height, g, height * g)
    #print("moist_nonstatic_energy:", moist_nonstatic_energy)

    guess_t = moist_nonstatic_energy / c_p
    guess_q = mpcalc.specific_humidity_from_dewpoint(pressure, guess_t)
    guess_mse = mpcalc.moist_static_energy(height, guess_t, guess_q).to("J/kg")

    #print("mse:", mse, "guess_mse:", guess_mse, "guess_t", guess_t)

    GUESS_CHANGE_COEF = 0.2
    while abs(mse.magnitude - guess_mse.magnitude) > 1:
        guess_diff = mse - guess_mse

        #print("mse:", mse, "guess_mse:", guess_mse, "guess_t", guess_t)

        guess_t -= -guess_diff / c_p * GUESS_CHANGE_COEF
        guess_q = mpcalc.specific_humidity_from_dewpoint(pressure, guess_t)
        guess_mse = mpcalc.moist_static_energy(height, guess_t, guess_q).to("J/kg")
    
    return guess_t

# test of iter solver

# test_T = 280 * units.kelvin
# test_p = 900 * units.hectopascal
# test_z = 1 * units.kilometer

# test_q = mpcalc.specific_humidity_from_dewpoint(test_p, test_T)

# print(test_T)
# print(test_q)

# test_mse = mpcalc.moist_static_energy(test_z, test_T, test_q)

# print(test_mse)

# retro_T = temperature_from_mse(test_mse, test_p, test_z)

# print(retro_T)

# test_z_interp = [0, 1000] * units.meter
# test_mse_interp = [300, 200] * units.kilojoule / units.kilogram
# test_z_input = 500 * units.meter

# z_interp_result = linear_interp(test_z_interp, test_mse_interp, test_z_input)

# print(z_interp_result)
