#
# AUTHOR: Amelia Urquhart (https://github.com/a-urq)
# VERSION: 1.1
# DATE: January 27, 2024
#

# USE PETERS ET AL 2022 LAPSE RATES INSTEAD OF MSE THING
# HAVE A SWITCH THAT CONTROLS ACCOUNTING FOR 

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
DRY_ADIABATIC_LAPSE_RATE: pint.Quantity = 9.761 * units.kelvin / units.kilometer
DEWPOINT_LAPSE_RATE: pint.Quantity = 1.8 * units.kelvin / units.kilometer

# Unlike the Java version, this expects arrays sorted in order of increasing height, decreasing pressure
# This is to keep in line with MetPy conventions
# Returns Tuple of { parcel_pressure, parcel_height, parcel_temperature, parcel_qv, parcel_qt }
@check_units("[pressure]", "[length]", "[temperature]", "[temperature]", "[speed]", "[speed]")
def calc_ecape_parcel(
        pressure: PintList, 
        height: PintList, 
        temperature: PintList, 
        dewpoint: PintList, 
        u_wind: PintList, 
        v_wind: PintList,
        align_to_input_pressure_values: bool,
        entrainment_switch: bool = True,
        pseudoadiabatic_switch: bool = True,
        cape_type: str = "most_unstable",
        storm_motion_type: str = "right_moving", 
        inflow_bottom: pint.Quantity = 0 * units.kilometer, 
        inflow_top: pint.Quantity = 1 * units.kilometer, 
        cape: pint.Quantity = None,
        el: pint.Quantity = None) -> Tuple[pint.Quantity, pint.Quantity, pint.Quantity, pint.Quantity, pint.Quantity]:
    
    if cape_type not in ['most_unstable', 'mixed_layer', 'surface_based']:
        sys.exit("Invalid 'cape_type' kwarg. Valid cape_types include ['most_unstable', 'mixed_layer', 'surface_based']")
    
    if storm_motion_type not in ['right_moving', 'left_moving', 'mean_wind']:
        sys.exit("Invalid 'storm_motion_type' kwarg. Valid storm_motion_types include ['right_moving', 'left_moving', 'mean_wind']")

    specific_humidity = []

    for i in range(len(pressure)):
        pressure_0 = pressure[i]
        dewpoint_0 = dewpoint[i]

        q_0 = specific_humidity_from_dewpoint(pressure_0, dewpoint_0).magnitude

        specific_humidity.append(q_0)

        # print("amelia q0: ", pressure_0, q_0)

    specific_humidity *= units('dimensionless')

    # print(specific_humidity)

    # moist_static_energy = mpcalc.moist_static_energy(height, temperature, specific_humidity).to("J/kg")

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
    # print("cape", cape)
        
    if cape <= 0:
        if align_to_input_pressure_values:
            pressure_raw = [None] * len(pressure)
            height_raw = [None] * len(pressure)
            temperature_raw = [None] * len(pressure)
            qv_raw = [None] * len(pressure)
            qt_raw = [None] * len(pressure)

            return (pressure_raw,  height_raw, temperature_raw, qv_raw, qt_raw)
        else:
            pressure_raw = [None]
            height_raw = [None]
            temperature_raw = [None]
            qv_raw = [None]
            qt_raw = [None]
            
            return (pressure_raw,  height_raw, temperature_raw, qv_raw, qt_raw)

    entr_rate = None

    if entrainment_switch:
        ecape = calc_ecape(height, pressure, temperature, specific_humidity, u_wind, v_wind, cape_type, cape)
        vsr = calc_sr_wind(pressure, u_wind, v_wind, height, storm_motion_type, inflow_bottom, inflow_top)
        storm_column_height = el - parcel_height

        cape = cape.to("joule / kilogram")
        ecape = ecape.to("joule / kilogram")
        vsr = vsr.to("meter / second")
        storm_column_height = storm_column_height.to("meter")

        print("amelia cape: ", cape)
        print("amelia ecape: ", ecape)
        print("amelia vsr: ", vsr)
        print("amelia storm_column_height: ", storm_column_height)

        r = updraft_radius(cape.magnitude, ecape.magnitude, vsr.magnitude, storm_column_height.magnitude)
        epsilon = entrainment_rate(r)

        print("ur inputs:", cape.magnitude, ecape.magnitude, vsr.magnitude, storm_column_height.magnitude)

        print("going into peters profile")
        print("ur:", r)
        print("eps:", epsilon)
        
        entr_rate = epsilon / units.meter
    else:
        entr_rate = 0 / units.meter

    #print("updr: ", r, " m")
    print("entr: ", entr_rate)

    parcel_temperature = parcel_temperature.to("degK")
    parcel_dewpoint = parcel_dewpoint.to("degK")

    parcel_qv = specific_humidity_from_dewpoint(parcel_pressure, parcel_dewpoint)
    parcel_qt = parcel_qv

    pressure_raw = []
    height_raw = []
    temperature_raw = []
    qv_raw = []
    qt_raw = []

    pressure_raw.append(parcel_pressure)
    height_raw.append(parcel_height)
    temperature_raw.append(parcel_temperature)
    qv_raw.append(parcel_qv)
    qt_raw.append(parcel_qt)

    parcel_moist_static_energy = mpcalc.moist_static_energy(parcel_height, parcel_temperature, parcel_qv)

    print("parcel z/q/q0:", parcel_height, parcel_qv, specific_humidity[0])
    print("specific humidity: ", specific_humidity)
    print("amelia entr rate:", entr_rate)

    prate = 1 / ECAPE_PARCEL_DZ
    if not pseudoadiabatic_switch:
        prate *= 0

    dqt_dz = 0 / ECAPE_PARCEL_DZ

    while parcel_pressure >= pressure[-1]:
        env_temperature = linear_interp(height, temperature, parcel_height)
        # parcel_pressure = pressure_at_height(parcel_pressure, ECAPE_PARCEL_DZ, env_temperature)
        # parcel_height += ECAPE_PARCEL_DZ

        parcel_saturation_qv = (1-parcel_qt)*r_sat(parcel_temperature,parcel_pressure,1)

        if parcel_saturation_qv > parcel_qv:
            parcel_pressure = pressure_at_height(parcel_pressure, ECAPE_PARCEL_DZ, env_temperature)
            parcel_height += ECAPE_PARCEL_DZ

            env_temperature = linear_interp(height, temperature, parcel_height)
            env_qv = linear_interp(height, specific_humidity, parcel_height)

            dT_dz = unsaturated_adiabatic_lapse_rate(parcel_temperature, parcel_qv, env_temperature, env_qv, entr_rate)
            dqv_dz = -entr_rate * (parcel_qv - env_qv)

            q_sat = specific_humidity_from_dewpoint(parcel_pressure, parcel_temperature)
            
            # print("amelia dT/dz:", dT_dz.m, parcel_temperature.m, parcel_qv.m, env_temperature.m, env_qv.m, parcel_pressure.m, entr_rate.m, "dqv_dz", dqv_dz.m, (-entr_rate * (parcel_qv - env_qv)).m, parcel_qv.m, env_qv.m)
            # print("amelia dT/dz:", dT_dz.m, parcel_temperature.m, parcel_qv.m, env_temperature.m, env_qv.m, parcel_pressure.m, parcel_height.m, entr_rate.m, "q_sat", q_sat)

            parcel_temperature += dT_dz * ECAPE_PARCEL_DZ
            parcel_qv += dqv_dz * ECAPE_PARCEL_DZ
            # parcel_qt += dqt_dz * ECAPE_PARCEL_DZ
            parcel_qt = parcel_qv

            # print("amelia qv:", parcel_qv)

            parcel_dewpoint = dewpoint_from_specific_humidity(parcel_pressure, parcel_qv)
        else:
            parcel_pressure = pressure_at_height(parcel_pressure, ECAPE_PARCEL_DZ, env_temperature)
            parcel_height += ECAPE_PARCEL_DZ

            env_temperature = linear_interp(height, temperature, parcel_height)
            env_qv = linear_interp(height, specific_humidity, parcel_height)

            dT_dz = None
            
            if pseudoadiabatic_switch:
                dT_dz = saturated_adiabatic_lapse_rate(parcel_temperature, parcel_qt, parcel_pressure, env_temperature, env_qv, entr_rate, prate, qt_entrainment=dqt_dz)
            else:
                dT_dz = saturated_adiabatic_lapse_rate(parcel_temperature, parcel_qt, parcel_pressure, env_temperature, env_qv, entr_rate, prate)

            new_parcel_qv = (1-parcel_qt)*r_sat(parcel_temperature, parcel_pressure, 1).to('kg/kg')

            if pseudoadiabatic_switch:
                dqt_dz = (new_parcel_qv - parcel_qv) / ECAPE_PARCEL_DZ
            else:
                dqt_dz = -entr_rate * (parcel_qt - env_qv) - prate * (parcel_qt - parcel_qv)

            if parcel_pressure < 40000 * units('Pa') and parcel_pressure > 20000 * units('Pa'):
                pass
                # print("amelia dT/dz:", dT_dz.m, parcel_temperature.m, parcel_qv.m, parcel_qt.m, parcel_pressure.m, entr_rate.m, prate.m, "dqt_dz", dqt_dz.m, (-entr_rate * (parcel_qt - env_qv)).m, parcel_qt.m, env_qv.m)
                # print("amelia new_parcel_qv:", new_parcel_qv, parcel_qt, )
                # print("amelia dT/dz:", dT_dz.m, parcel_temperature.m, parcel_qv.m, parcel_qt.m, parcel_pressure.m, entr_rate.m, prate.m, "dqt_dz", dqt_dz.m, -entr_rate * (parcel_qt - env_qv), parcel_qt, env_qv)
                # print("amelia dT/dz:", dT_dz.m, parcel_qt.m, parcel_pressure.m, "dqt_dz", dqt_dz.m, (-entr_rate * (parcel_qt - env_qv)).m,  (-prate * (parcel_qt - parcel_qv)).m, parcel_qt.m, env_qv.m, prate.m)
                # print("amelia dqt_dz:", dqt_dz, parcel_pressure)
                # print("amelia qv qt qv0:", parcel_qv, parcel_qt, env_qv)
            # print("dqt_dz - 1:", -entr_rate * (parcel_qv - env_qv))
            # print("dqt_dz - 2:", entr_rate)
            # print("dqt_dz - 3:", (parcel_qv))
            # print("dqt_dz - 4:", (env_qv))
            # print("dqt_dz - 5:", (parcel_qv - env_qv))
            # print("dqt_dz - 6:", - prate * (parcel_qt - parcel_qv))
            # print("dqt_dz - 7:", -prate)
            # print("dqt_dz - 8:", (parcel_qt - parcel_qv))

            parcel_temperature += dT_dz * ECAPE_PARCEL_DZ
            parcel_qv = new_parcel_qv

            if pseudoadiabatic_switch:
                parcel_qt = parcel_qv
            else:
                dqt_dz = -entr_rate * (parcel_qt - env_qv) - prate * (parcel_qt - parcel_qv)
                parcel_qt += dqt_dz * ECAPE_PARCEL_DZ

            # print("qv:", parcel_qv)
            # print("qt & dqt:", parcel_qt, dqt_dz)

            if parcel_qt < parcel_qv:
                parcel_qv = parcel_qt

        pressure_raw.append(parcel_pressure)
        height_raw.append(parcel_height)
        temperature_raw.append(parcel_temperature)
        qv_raw.append(parcel_qv)
        qt_raw.append(parcel_qt)

        # print(parcel_pressure, parcel_height, parcel_temperature, parcel_qv, parcel_qt)

    # for i in range(len(qv_raw)):
    #     print("amelia q profile:", pressure_raw[i], qv_raw[i], qt_raw[i])

    pressure_units = pressure_raw[-1].units
    height_units = height_raw[-1].units
    temperature_units = temperature_raw[-1].units
    qv_units = qv_raw[-1].units
    qt_units = qt_raw[-1].units

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
    qv_nondim = [dewpoint.magnitude for dewpoint in qv_raw]
    qt_nondim = [dewpoint.magnitude for dewpoint in qt_raw]

    # makes it work ok with sounderpy
    if align_to_input_pressure_values:
        pressure_nondim_aligned = []
        height_nondim_aligned = []
        temperature_nondim_aligned = []
        qv_nondim_aligned = []
        qt_nondim_aligned = []

        #print(temperature_nondim)
        #print(dewpoint_nondim)
        #print(linear_interp(height, temperature_nondim, 1000 * units.meter))
        #print("above is debugging lin interp")

        for i in range(len(height)):
            input_pressure = pressure[i]
            input_height = height[i]

            # print("searching for interp_t at height", input_height)

            new_t = rev_linear_interp(pressure_raw, temperature_nondim, input_pressure)
            new_qv = rev_linear_interp(pressure_raw, qv_nondim, input_pressure)
            new_qt = rev_linear_interp(pressure_raw, qt_nondim, input_pressure)

            pressure_nondim_aligned.append(input_pressure.magnitude)
            height_nondim_aligned.append(input_height.magnitude)
            temperature_nondim_aligned.append(new_t)
            qv_nondim_aligned.append(new_qv)
            qt_nondim_aligned.append(new_qt)

        pressure_nondim = pressure_nondim_aligned
        height_nondim = height_nondim_aligned
        temperature_nondim = temperature_nondim_aligned
        qv_nondim = qv_nondim_aligned
        qt_nondim = qt_nondim_aligned

    pressure_qty : pint.Quantity = pressure_nondim * pressure_units
    height_qty : pint.Quantity = height_nondim * height_units
    temperature_qty : pint.Quantity = temperature_nondim * temperature_units
    qv_qty : pint.Quantity = qv_nondim * qv_units
    qt_qty : pint.Quantity = qt_nondim * qt_units

    print(pressure_qty[0], height_qty[0], temperature_qty[0], qv_qty[0], qt_qty[0])

    return ( pressure_qty, height_qty, temperature_qty, qv_qty, qt_qty )

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

c_pd = 1005 * units('J/kg') / units('K')
c_pv = 1875 * units('J/kg') / units('K')
def specific_heat_capacity_of_moist_air(specific_humidity: pint.Quantity):
    c_p = specific_humidity * c_pv + (1 - specific_humidity) * c_pd

    return c_p

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
def temperature_from_mse(mse: pint.Quantity, pressure: pint.Quantity, height: pint.Quantity, specific_humidity: pint.Quantity) -> pint.Quantity:
    #print(mse)
    mse = mse.to("J/kg")
    #print(mse)

    moist_nonstatic_energy = mse - height * g # made up parameter lol

    #print(height, g, height * g)
    #print("moist_nonstatic_energy:", moist_nonstatic_energy)

    c_pm = specific_heat_capacity_of_moist_air(specific_humidity)

    guess_t = moist_nonstatic_energy / c_pm
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

def specific_humidity_from_dewpoint(pressure, dewpoint):
	vapor_pressure_ = vapor_pressure(dewpoint)

	return specific_humidity(pressure, vapor_pressure_) * units('dimensionless')

def vapor_pressure(dewpoint):
    dewpoint_nondim = dewpoint.to('K').magnitude
	
    e0 = 611 * units('Pa')
    t0 = 273.15
	
    return e0 * math.exp(latent_heat_of_vaporization / water_vapor_gas_constant * (1 / t0 - 1 / dewpoint_nondim))

def specific_humidity(pressure: pint.Quantity, vapor_pressure: pint.Quantity) -> pint.Quantity:
    pressure_nondim = pressure.to('Pa').magnitude
    vapor_pressure_nondim = vapor_pressure.to('Pa').magnitude

    water_vapor_density = absolute_humidity(vapor_pressure_nondim, 280); # kg m^-3
    air_density = dry_air_density(pressure_nondim - vapor_pressure_nondim, 280); # kg m^-3

    # print("d_wv:", water_vapor_density)
    # print("d_da:", air_density)

    return water_vapor_density / (water_vapor_density + air_density)

dry_air_gas_constant = 287
water_vapor_gas_constant = 461.5
latent_heat_of_vaporization = 2500000

def absolute_humidity(vapor_pressure, temperature):
	water_vapor_density = vapor_pressure / (water_vapor_gas_constant * temperature)

	return water_vapor_density

def dry_air_density(dry_air_pressure, temperature):
	dry_air_density = dry_air_pressure / (dry_air_gas_constant * temperature)

	return dry_air_density

def dewpoint_from_specific_humidity(pressure, specific_humidity):
    vapor_pressure = vapor_pressure_from_specific_humidity(pressure.to('Pa').magnitude, specific_humidity)
    dewpoint = dewpoint_from_vapor_pressure(vapor_pressure)
    return dewpoint

def vapor_pressure_from_specific_humidity(pressure, specific_humidity):
    water_vapor_gas_constant = 461.5  # J/(kg·K)
    dry_air_gas_constant = 287  # J/(kg·K)

    numerator = specific_humidity * pressure
    denominator_term = (1 / water_vapor_gas_constant + specific_humidity / dry_air_gas_constant
                        - specific_humidity / water_vapor_gas_constant)

    vapor_pressure = numerator / (dry_air_gas_constant * denominator_term)

    return vapor_pressure

def dewpoint_from_vapor_pressure(vapor_pressure):
    e0 = 611  # Pascals
    t0 = 273.15  # Kelvins
    latent_heat_of_vaporization = 2.5e6  # J/kg

    vapor_pres_nondim = vapor_pressure

    # print(1 / t0)
    # print((461.5 / latent_heat_of_vaporization))
    # print(vapor_pressure)
    # print(e0)
    # print(math.log(vapor_pres_nondim / e0))

    dewpoint_reciprocal = 1 / t0 - (461.5 / latent_heat_of_vaporization) * math.log(vapor_pres_nondim / e0)

    return (1 / dewpoint_reciprocal) * units('K')

g = 9.81 * units("m")/units("s")/units("s")
c_pd = 1005 * units("J/kg")/units("K")
c_pv = 1870 * units("J/kg")/units("K")
c_pl = 4190 * units("J/kg")/units("K")
c_pi = 2106 * units("J/kg")/units("K")
R_d = 287.04 * units("J/kg")/units("K")
R_v = 461.5 * units("J/kg")/units("K")
L_v_trip = 2501000 * units("J/kg")
L_i_trip = 333000 * units("J/kg")
T_trip = 273.15 * units("K")

phi = R_d/R_v

@check_units('[temperature]', '[dimensionless]', '[dimensionless]')
def density_temperature(temperature, qv, qt) -> pint.Quantity:
    t_rho = temperature * (1 - qt + qv/phi)

    return t_rho

# Equation 19 in Peters et. al. 2022 (https://journals.ametsoc.org/view/journals/atsc/79/3/JAS-D-21-0118.1.xml)
@check_units('[temperature]', '[dimensionless]', '[temperature]', '[dimensionless]')
def unsaturated_adiabatic_lapse_rate(temperature_parcel: pint.Quantity, qv_parcel: pint.Quantity, temperature_env: pint.Quantity, qv_env: pint.Quantity, entrainment_rate: pint.Quantity) -> pint.Quantity:
    temperature_entrainment = -entrainment_rate * (temperature_parcel - temperature_env)
    
    density_temperature_parcel = density_temperature(temperature_parcel, qv_parcel, qv_parcel)
    density_temperature_env = density_temperature(temperature_env, qv_env, qv_env)

    buoyancy = g * (density_temperature_parcel - density_temperature_env)/density_temperature_env

    c_pmv = (1 - qv_parcel) * c_pd + qv_parcel * c_pv

    # print("amelia cpmv:", c_pmv)
    # print("amelia B:", buoyancy)
    # print("amelia eps:", temperature_entrainment)

    term_1 = -g/c_pd
    term_2 = 1 + (buoyancy/g)
    term_3 = c_pmv/c_pd

    dTdz = term_1 * (term_2/term_3) + temperature_entrainment

    return dTdz

@check_units('[temperature]', '[temperature]', '[temperature]')
def ice_fraction(temperature, warmest_mixed_phase_temp, coldest_mixed_phase_temp):
    if (temperature >= warmest_mixed_phase_temp):
        return 0
    elif (temperature <= coldest_mixed_phase_temp):
        return 1
    else:
        return (1/(coldest_mixed_phase_temp - warmest_mixed_phase_temp))*(temperature - warmest_mixed_phase_temp)
    
@check_units('[temperature]', '[temperature]', '[temperature]')
def ice_fraction_deriv(temperature, warmest_mixed_phase_temp, coldest_mixed_phase_temp):
    if (temperature >= warmest_mixed_phase_temp):
        return 0 / units('K')
    elif (temperature <= coldest_mixed_phase_temp):
        return 0 / units('K')
    else:
        return (1/(coldest_mixed_phase_temp - warmest_mixed_phase_temp))
    
vapor_pres_ref = 611.2 * units("Pa")
# borrowed and adapted from ECAPE_FUNCTIONS
def r_sat(temperature, pressure, ice_flag: int, warmest_mixed_phase_temp: pint.Quantity = 273.15 * units("K"), coldest_mixed_phase_temp: pint.Quantity = 253.15 * units("K")):
        if ice_flag == 2:
            term_1=(c_pv - c_pi)/R_v
            term_2=(L_v_trip - T_trip * (c_pv - c_pi))/R_v
            esi=np.exp((temperature - T_trip)*term_2/(temperature*T_trip))*vapor_pres_ref*(temperature/T_trip)**(term_1)
            q_sat=phi * esi/(pressure - esi)

            return q_sat
        elif ice_flag == 1:
            omega = ice_fraction(temperature, warmest_mixed_phase_temp, coldest_mixed_phase_temp)

            qsat_l = r_sat(temperature, pressure, 0)
            qsat_i = r_sat(temperature, pressure, 2)

            q_sat=(1-omega)*qsat_l + (omega)*qsat_i

            return q_sat
        else:
            term_1=(c_pv - c_pl)/R_v
            term_2=(L_v_trip - T_trip * (c_pv - c_pl))/R_v
            esi=np.exp((temperature - T_trip)*term_2/(temperature*T_trip))*vapor_pres_ref*(temperature/T_trip)**(term_1)
            q_sat=phi * esi/(pressure - esi)

            return q_sat

# Equation 24 in Peters et. al. 2022 (https://journals.ametsoc.org/view/journals/atsc/79/3/JAS-D-21-0118.1.xml)
# @check_units('[temperature]', '[dimensionless]',  '[dimensionless]', '[temperature]', '[dimensionless]', '[dimensionless]')
def saturated_adiabatic_lapse_rate(temperature_parcel: pint.Quantity, qt_parcel: pint.Quantity, pressure_parcel: pint.Quantity, temperature_env: pint.Quantity, qv_env: pint.Quantity, entrainment_rate: pint.Quantity, prate: pint.Quantity, warmest_mixed_phase_temp: pint.Quantity = 273.15 * units("K"), coldest_mixed_phase_temp: pint.Quantity = 253.15 * units("K"), qt_entrainment: pint.Quantity = None) -> pint.Quantity:
    omega = ice_fraction(temperature_parcel, warmest_mixed_phase_temp, coldest_mixed_phase_temp)
    d_omega = ice_fraction_deriv(temperature_parcel, warmest_mixed_phase_temp, coldest_mixed_phase_temp)

    q_vsl = (1 - qt_parcel)*r_sat(temperature_parcel, pressure_parcel, 0)
    q_vsi = (1 - qt_parcel)*r_sat(temperature_parcel, pressure_parcel, 2)
    
    qv_parcel = (1 - omega) * q_vsl + omega * q_vsi

    temperature_entrainment = -entrainment_rate * (temperature_parcel - temperature_env)
    qv_entrainment = -entrainment_rate * (qv_parcel - qv_env)

    if qt_entrainment == None:
        qt_entrainment = -entrainment_rate * (qt_parcel - qv_env) - prate * (qt_parcel - qv_parcel)

    # print("dqv_dz:", qv_entrainment)

    # print("amelia eps_T:", temperature_entrainment)
    # print("amelia eps_qv:", qv_entrainment)
    # print("amelia eps_qt:", qt_entrainment)
    q_condensate = qt_parcel - qv_parcel
    ql_parcel = q_condensate * (1 - omega)
    qi_parcel = q_condensate * omega

    c_pm = (1 - qt_parcel) * c_pd + qv_parcel * c_pv + ql_parcel * c_pl + qi_parcel * c_pi
    
    density_temperature_parcel = density_temperature(temperature_parcel, qv_parcel, qt_parcel)
    density_temperature_env = density_temperature(temperature_env, qv_env, qv_env)

    buoyancy = g * (density_temperature_parcel - density_temperature_env)/density_temperature_env

    # print("density_temperature_parcel:", density_temperature_parcel)
    # print("density_temperature_env:", density_temperature_env)

    L_v = L_v_trip + (temperature_parcel - T_trip)*(c_pv - c_pl)
    L_i = L_i_trip + (temperature_parcel - T_trip)*(c_pl - c_pi)

    L_s = L_v + omega * L_i

    Q_vsl = q_vsl/(phi - phi*qt_parcel + qv_parcel)
    Q_vsi = q_vsi/(phi - phi*qt_parcel + qv_parcel)
    # print("q_vsl:", q_vsl)
    # print("q_vsi:", q_vsi)
    # print("Q_vsl:", Q_vsl)
    # print("Q_vsi:", Q_vsi)

    Q_M = (1 - omega) * (q_vsl)/(1 - Q_vsl) + omega * (q_vsi)/(1 - Q_vsi)
    L_M = (1 - omega) * L_v * (q_vsl)/(1 - Q_vsl) + omega * (L_v + L_i) * (q_vsi)/(1 - Q_vsi)
    R_m0 = (1 - qv_env) * R_d + qv_env * R_v

    term_1 = buoyancy
    term_2 = g
    term_3 = ((L_s * Q_M) / (R_m0 * temperature_env)) * g
    term_4 = (c_pm - L_i * (qt_parcel - qv_parcel) * d_omega) * temperature_entrainment
    term_5 = L_s * (qv_entrainment + qv_parcel/(1-qt_parcel) * qt_entrainment) # - (q_vsi - q_vsl) * d_omega) # peters left this end bit out

    term_6 = c_pm
    term_7 = (L_i * (qt_parcel - qv_parcel) - L_s * (q_vsi - q_vsl)) * d_omega
    term_8 = (L_s * L_M)/(R_v * temperature_parcel * temperature_parcel)

    # print("amelia 4.1: ", (c_pm - L_i * (qt_parcel - qv_parcel) * d_omega) * temperature_entrainment)
    # print("amelia 4.2: ", (c_pm - L_i * (qt_parcel - qv_parcel) * d_omega))
    # print("amelia 4.3: ", (c_pm))
    # print("amelia 4.4: ", (L_i * (qt_parcel - qv_parcel) * d_omega))
    # print("amelia 4.5: ", (L_i))
    # print("amelia 4.6: ", ((qt_parcel)))
    # print("amelia 4.7: ", ((qv_parcel)))
    # print("amelia 4.8: ", ((qt_parcel - qv_parcel)))
    # print("amelia 4.9: ", (d_omega))

    # print("term 1:", term_1)
    # print("term 2:", term_2)
    # print("term 3:", term_3)
    # print("term 4:", term_4)
    # print("term 5:", term_5)
    # print("term 6:", term_6)
    # print("term 7:", term_7)
    # print("term 8:", term_8)
    # print("numerator:", -(term_1 + term_2 + term_3 - term_4 - term_5))
    # print("denominator:", (term_6 - term_7 + term_8))
    # print("entrainment:", entrainment_rate)
    # print("T entrainment:", temperature_entrainment)

    return -(term_1 + term_2 + term_3 - term_4 - term_5) / (term_6 - term_7 + term_8)


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

# c_p = specific_heat_capacity_of_moist_air(0.0137)
# print(c_p)

# T = 300 * units('K')
# qt_parcel = 0.001 * units('dimensionless')
# p = 87500 * units('Pa')
# T0 = 300 * units('K')
# qv0 = 0.001 * units('dimensionless')
# entr = 0 / units('m')
# prate = 0 / units('m')

# gamma_m = saturated_adiabatic_lapse_rate(T, qt_parcel, p, T0, qv0, entr, prate).to("K/km")
# print(gamma_m)

# gamma_m = saturated_adiabatic_lapse_rate(T, 0*qt_parcel, p, T0, qv0, entr, prate).to("K/km")
# print(gamma_m)

# gamma_m = saturated_adiabatic_lapse_rate(T, 40*qt_parcel, p, T0, qv0, entr, prate).to("K/km")
# print(gamma_m)

# T = 275 * units('K')
# qt_parcel = 0.001 * units('dimensionless')
# p = 87500 * units('Pa')
# T0 = 275 * units('K')
# qv0 = 0.001 * units('dimensionless')
# entr = 0 / units('m')
# prate = 0 / units('m')

# gamma_m = saturated_adiabatic_lapse_rate(T, qt_parcel, p, T0, qv0, entr, prate).to("K/km")
# print(gamma_m)

T = 252.6539 * units('K')
qt_parcel = 0.011993985 * units('dimensionless')
p = 34553 * units('Pa')
T0 = 243.15 * units('K')
qv0 = 0.000208959 * units('dimensionless')
entr = 5.02522e-05 / units('m')
prate = 0.00 / units('m')
prate_2 = 0.004 / units('m')
T1 = 273.15 * units('K')
T2 = 253.15 * units('K')

# print("q-parcel-pseudo: ", specific_humidity_from_dewpoint(p, T))

gamma_m = saturated_adiabatic_lapse_rate(T, qt_parcel, p, T0, qv0, entr, prate, T1, T2).to("K/km")
print(gamma_m)

from ECAPE_FUNCTIONS import moislif, drylift

q_vsl = (1 - qt_parcel)*r_sat(T, p, 0)
q_vsi = (1 - qt_parcel)*r_sat(T, p, 2)
omega = ice_fraction(T, T1, T2)
    
qv_parcel = (1 - omega) * q_vsl + omega * q_vsi

gamma_m = moislif(T.magnitude, qv_parcel.magnitude, q_vsl.magnitude, q_vsi.magnitude, p.magnitude, T0.magnitude, qv0.magnitude, qt_parcel.magnitude, entr.magnitude, prate.magnitude, T1.magnitude, T2.magnitude)
print(gamma_m * 1000)

# test_pressure = 85000 * units('Pa')
# for i in range(0, 100, 1):
#     test_dewpoint = (i + 223.15) * units('K')
    
#     amelia_q0 = specific_humidity_from_dewpoint(test_pressure, test_dewpoint)
#     metpy_q0 = mpcalc.specific_humidity_from_dewpoint(test_pressure, test_dewpoint)

#     print(test_pressure, test_dewpoint, amelia_q0, metpy_q0, amelia_q0 - metpy_q0, amelia_q0 / metpy_q0)

# test_pressure = 25000 * units('Pa')
# for i in range(0, 100, 1):
#     test_dewpoint = (i + 223.15) * units('K')
    
#     amelia_q0 = specific_humidity_from_dewpoint(test_pressure, test_dewpoint)
#     metpy_q0 = mpcalc.specific_humidity_from_dewpoint(test_pressure, test_dewpoint)

#     print(test_pressure, test_dewpoint, amelia_q0, metpy_q0, amelia_q0 - metpy_q0, amelia_q0 / metpy_q0)

# prate_2 = 0.01 / units('m')

# gamma_m = moislif(T.magnitude, qv_parcel.magnitude, q_vsl.magnitude, q_vsi.magnitude, p.magnitude, T0.magnitude, qv0.magnitude, qt_parcel.magnitude, entr.magnitude, prate_2.magnitude, T1.magnitude, T2.magnitude)
# print(gamma_m * 1000)

# prate_2 = 0.004 / units('m')

# gamma_m = moislif(T.magnitude, qv_parcel.magnitude, q_vsl.magnitude, q_vsi.magnitude, p.magnitude, T0.magnitude, qv0.magnitude, qt_parcel.magnitude, entr.magnitude, prate_2.magnitude, T1.magnitude, T2.magnitude)
# print(gamma_m * 1000)

# prate_2 = 0.002 / units('m')

# gamma_m = moislif(T.magnitude, qv_parcel.magnitude, q_vsl.magnitude, q_vsi.magnitude, p.magnitude, T0.magnitude, qv0.magnitude, qt_parcel.magnitude, entr.magnitude, prate_2.magnitude, T1.magnitude, T2.magnitude)
# print(gamma_m * 1000)

# gamma_d = unsaturated_adiabatic_lapse_rate(T, qv0, T0, qv0, entr).to("K/km")
# print(gamma_d)

# print("d_omega:", ice_fraction_deriv(263.15 * units('K'), 273.15 * units('K'), 253.15 * units('K')))
    
#gamma_d
    
# T = 298.08788821800584 * units('K')
# q = 0.0163356632155656 * units('dimensionless') 
# T0 = 298.904907537339 * units('K') 
# q0 = 0.015581996527348846 * units('dimensionless') 
# p = 949.047328986073 * units('hPa') 
# entr_rate = 5.0252220440967176e-05 / units('m')
# # entr_rate = 0 / units('m')

# gamma_d = unsaturated_adiabatic_lapse_rate(T, q,T0, q0, entr_rate).to('K/km')

# print("amelia gamma_d: ", gamma_d)

# gamma_d = drylift(T.m, q.m, T0.m, q0.m, entr_rate.m)

# print("peters gamma_d: ", gamma_d * 1000)


# T = 298.08788821800584 * units('K')
# p = 949.047328986073 * units('hPa') 

# q_sat = specific_humidity_from_dewpoint(p, T)

# print("vapor_pressure:", vapor_pressure(T))
# print("q_sat:", q_sat)

# q_sat = mpcalc.specific_humidity_from_dewpoint(p, T)

# print("vapor_pressure:", mpcalc.saturation_vapor_pressure(T))
# print("q_sat:", q_sat)