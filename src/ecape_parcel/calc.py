#
# AUTHOR: Amelia Urquhart (https://github.com/a-urq)
# VERSION: 1.2.2
# DATE: March 25, 2024
#

import metpy as mpy
import metpy.calc as mpcalc
import pint
import math
import numpy as np
import sys

from ecape_parcel.ecape_calc import calc_ecape_ncape, calc_sr_wind
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

# @param updraftRadius              Units: Meters
# @return entrainment_rate:         Units: m^-1
def entrainment_rate(cape: float, ecape: float, ncape: float, vsr: float, storm_column_height: float) -> float:
    E_A_tilde = ecape/cape
    N_tilde = ncape/cape
    vsr_tilde = vsr/np.sqrt(2*cape)

    E_tilde = E_A_tilde - vsr_tilde**2

    entrainment_rate = (2 * (1 - E_tilde)/(E_tilde + N_tilde))/(storm_column_height)

    return entrainment_rate

def updraft_radius(entrainment_rate: float) -> float:
    updraft_radius = np.sqrt(2 * k2 * L_mix / (Pr * entrainment_rate))

    return updraft_radius

ECAPE_PARCEL_DZ: pint.Quantity = 20 * units.meter

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
        mixed_layer_depth_pressure: pint.Quantity = 100 * units("hPa"),
        mixed_layer_depth_height: pint.Quantity = None,
        storm_motion_type: str = "right_moving", 
        inflow_layer_bottom: pint.Quantity = 0 * units.kilometer, 
        inflow_layer_top: pint.Quantity = 1 * units.kilometer, 
        cape: pint.Quantity = None,
        lfc: pint.Quantity = None,
        el: pint.Quantity = None,
        storm_motion_u: pint.Quantity = None,
        storm_motion_v: pint.Quantity = None,
        origin_pressure: pint.Quantity = None,
        origin_height: pint.Quantity = None,
        origin_temperature: pint.Quantity = None,
        origin_dewpoint: pint.Quantity = None) -> Tuple[pint.Quantity, pint.Quantity, pint.Quantity, pint.Quantity, pint.Quantity]:
    
    if cape_type not in ['most_unstable', 'mixed_layer', 'surface_based', 'user_defined']:
        sys.exit("Invalid 'cape_type' kwarg. Valid cape_types include ['most_unstable', 'mixed_layer', 'surface_based', 'user_defined']")
    
    if storm_motion_type not in ['right_moving', 'left_moving', 'mean_wind', 'user_defined']:
        sys.exit("Invalid 'storm_motion_type' kwarg. Valid storm_motion_types include ['right_moving', 'left_moving', 'mean_wind', 'user_defined']")

    specific_humidity = []

    for i in range(len(pressure)):
        pressure_0 = pressure[i]
        dewpoint_0 = dewpoint[i]

        q_0 = mpcalc.specific_humidity_from_dewpoint(pressure_0, dewpoint_0).magnitude

        specific_humidity.append(q_0)

        # print("amelia q0: ", pressure_0, q_0)

    specific_humidity *= units('dimensionless')

    # print(specific_humidity)

    # moist_static_energy = mpcalc.moist_static_energy(height, temperature, specific_humidity).to("J/kg")

    parcel_pressure = -1024
    parcel_height = -1024
    parcel_temperature = -1024
    parcel_dewpoint = -1024
    
    # have a "user_defined" switch option
    if "user_defined" == cape_type:
        if origin_pressure != None:
            parcel_pressure = origin_pressure
        else:
            parcel_pressure = pressure[0]
        
        if origin_height != None:
            parcel_height = origin_height
        else:
            parcel_height = height[0]

        if origin_temperature != None:
            parcel_temperature = origin_temperature
        else:
            parcel_temperature = temperature[0]

        if origin_dewpoint != None:
            parcel_dewpoint = origin_dewpoint
        else:
            parcel_dewpoint = dewpoint[0]
    elif "most_unstable" == cape_type:
        parcel_pressure, parcel_temperature, parcel_dewpoint, mu_idx = mpcalc.most_unstable_parcel(pressure, temperature, dewpoint)
        parcel_height = height[mu_idx]
    elif "mixed_layer" == cape_type:
        env_potential_temperature = mpcalc.potential_temperature(pressure, temperature)
        env_specific_humidity = mpcalc.specific_humidity_from_dewpoint(pressure, dewpoint)

        env_idxs_to_include_in_average = None

        if mixed_layer_depth_pressure != None:
            mixed_layer_top_pressure = pressure[0] - mixed_layer_depth_pressure
            env_idxs_to_include_in_average = np.where(pressure >= mixed_layer_top_pressure)[0]
        elif mixed_layer_depth_height != None:
            mixed_layer_top_height = height[0] + mixed_layer_depth_height
            env_idxs_to_include_in_average = np.where(height <= mixed_layer_top_height)[0]
            pass
        else:
            mixed_layer_depth_pressure = 100 * units("hPa")
            mixed_layer_top_pressure = pressure[0] - mixed_layer_depth_pressure
            env_idxs_to_include_in_average = np.where(pressure >= mixed_layer_top_pressure)[0]

        avg_potential_temperature_sum = 0.0
        avg_specific_humidity_sum = 0.0
        for i in range(len(env_idxs_to_include_in_average)):
            avg_potential_temperature_sum += env_potential_temperature[env_idxs_to_include_in_average[i]]
            avg_specific_humidity_sum += env_specific_humidity[env_idxs_to_include_in_average[i]]

        avg_potential_temperature = avg_potential_temperature_sum / len(env_idxs_to_include_in_average)
        avg_specific_humidity = avg_specific_humidity_sum / len(env_idxs_to_include_in_average)

        parcel_pressure = pressure[0]
        parcel_height = height[0]
        parcel_temperature = mpcalc.temperature_from_potential_temperature(parcel_pressure, avg_potential_temperature)
        parcel_dewpoint = mpcalc.dewpoint_from_specific_humidity(parcel_pressure, parcel_temperature, avg_specific_humidity)
    elif "surface_based" == cape_type:
        parcel_pressure = pressure[0]
        parcel_height = height[0]
        parcel_temperature = temperature[0]
        parcel_dewpoint = dewpoint[0]
    else:
        parcel_pressure = pressure[0]
        parcel_height = height[0]
        parcel_temperature = temperature[0]
        parcel_dewpoint = dewpoint[0]
        
    # print("in house cape/el calc:", cape, el, entrainment_switch)
    if (cape == None or lfc == None or el == None) and entrainment_switch == True:
        # print("-- using in-house cape --")
        undiluted_parcel_profile = calc_ecape_parcel(pressure, height, temperature, dewpoint, u_wind, v_wind, align_to_input_pressure_values, False, pseudoadiabatic_switch, cape_type, mixed_layer_depth_pressure, mixed_layer_depth_height, storm_motion_type, inflow_layer_bottom, inflow_layer_top, origin_pressure=origin_pressure, origin_height=origin_height, origin_temperature=origin_temperature, origin_dewpoint=origin_dewpoint)

        undiluted_parcel_profile_z = undiluted_parcel_profile[1]
        undiluted_parcel_profile_T = undiluted_parcel_profile[2]
        undiluted_parcel_profile_qv = undiluted_parcel_profile[3]
        undiluted_parcel_profile_qt = undiluted_parcel_profile[4]

        undil_cape, _, undil_lfc, undil_el = custom_cape_cin_lfc_el(undiluted_parcel_profile_z, undiluted_parcel_profile_T, undiluted_parcel_profile_qv, undiluted_parcel_profile_qt, height, temperature, specific_humidity)
        # cape, _ = mpcalc.cape_cin(pressure, temperature, dewpoint, parcel_profile)
        # el = calc_el_height(pressure, height, temperature, dewpoint, parcel_func[cape_type])[1]

        if cape == None:
            cape = undil_cape

        if lfc == None:
            lfc = undil_lfc

        if el == None:
            el = undil_el

    # print("el: ", el)
    # print("parcel_height: ", parcel_height)
        
    # print(height)
    # print(pressure)
    # print(temperature)
    # print(specific_humidity)
    # print(u_wind)
    # print(v_wind)
    # print(cape_type)
    # print("post in-house cape:", cape)
        
    if entrainment_switch == True:
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
        # print("amelia calc ecape")
        # print("lfc:", lfc)
        # print("el:", el)
        ecape, ncape = calc_ecape_ncape(height, pressure, temperature, specific_humidity, u_wind, v_wind, cape_type, cape, inflow_bottom=inflow_layer_bottom, inflow_top=inflow_layer_top, storm_motion=storm_motion_type, lfc=lfc, el=el, u_sm=storm_motion_u, v_sm=storm_motion_v)
        vsr = calc_sr_wind(pressure, u_wind, v_wind, height, inflow_layer_bottom, inflow_layer_top, storm_motion_type, sm_u=storm_motion_u, sm_v=storm_motion_v)
        storm_column_height = el - parcel_height

        cape = cape.to("joule / kilogram")
        ecape = ecape.to("joule / kilogram")
        ncape = ncape.to("joule / kilogram")
        vsr = vsr.to("meter / second")
        storm_column_height = storm_column_height.to("meter")

        # print("amelia ecape env profile")
        # for i in range(len(height)):
        #     print(height[i], pressure[i], temperature[i].to('degK'), specific_humidity[i], u_wind[i], v_wind[i])

        # print("amelia cape: ", cape)
        # print("amelia eil0: ", inflow_layer_bottom)
        # print("amelia eil1: ", inflow_layer_top)
        # print("amelia psi: ", calc_psi(storm_column_height))
        # print("amelia ecape: ", ecape)
        # print("amelia vsr: ", vsr)
        # print("amelia storm_column_height: ", storm_column_height)

        epsilon = entrainment_rate(cape.magnitude, ecape.magnitude, ncape.magnitude, vsr.magnitude, storm_column_height.magnitude)

        # print("amelia ur inputs:", cape.magnitude, ecape.magnitude, vsr.magnitude, storm_column_height.magnitude)
        # print("amelia ur:", r)
        # print("amelia eps:", epsilon)
        
        entr_rate = epsilon / units.meter
    else:
        entr_rate = 0 / units.meter

    #print("updr: ", r, " m")
    # print("entr: ", entr_rate)

    parcel_temperature = parcel_temperature.to("degK")
    parcel_dewpoint = parcel_dewpoint.to("degK")

    parcel_qv = mpcalc.specific_humidity_from_dewpoint(parcel_pressure, parcel_dewpoint)
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

    # print("parcel z/q/q0:", parcel_height, parcel_qv, specific_humidity[0])
    # print("specific humidity: ", specific_humidity)
    # print("amelia entr rate:", entr_rate)

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

            # q_sat = specific_humidity_from_dewpoint(parcel_pressure, parcel_temperature)
            
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

    pressure_nondim = [None] * len(pressure_raw)
    height_nondim = [None] * len(height_raw)
    temperature_nondim = [None] * len(temperature_raw)
    qv_nondim = [None] * len(qv_raw)
    qt_nondim = [None] * len(qt_raw)

    for i in range(len(height_raw)):
        pressure_nondim[i] = pressure_raw[i].magnitude
        height_nondim[i] = height_raw[i].magnitude
        temperature_nondim[i] = temperature_raw[i].magnitude
        qv_nondim[i] = qv_raw[i].magnitude
        qt_nondim[i] = qt_raw[i].magnitude

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

            if input_height >= height_raw[0]:
                new_t = rev_linear_interp(pressure_raw, temperature_nondim, input_pressure)
                new_qv = rev_linear_interp(pressure_raw, qv_nondim, input_pressure)
                new_qt = rev_linear_interp(pressure_raw, qt_nondim, input_pressure)

                pressure_nondim_aligned.append(input_pressure.magnitude)
                height_nondim_aligned.append(input_height.magnitude)
                temperature_nondim_aligned.append(new_t)
                qv_nondim_aligned.append(new_qv)
                qt_nondim_aligned.append(new_qt)
            else:
                pressure_nondim_aligned.append(input_pressure.magnitude)
                height_nondim_aligned.append(input_height.magnitude)
                temperature_nondim_aligned.append(temperature[i].to("degK").magnitude)
                qv_nondim_aligned.append(specific_humidity[i].magnitude)
                qt_nondim_aligned.append(specific_humidity[i].magnitude)

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

    # print(pressure_qty[0], height_qty[0], temperature_qty[0], qv_qty[0], qt_qty[0])

    return ( pressure_qty, height_qty, temperature_qty, qv_qty, qt_qty )

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

# allows for true adiabatic CAPE calculation and also bypasses metpy weirdness even for pseudoadiabatic
# only intended for undiluted
def custom_cape_cin_lfc_el(
        parcel_height: pint.Quantity,
        parcel_temperature: pint.Quantity,
        parcel_qv: pint.Quantity,
        parcel_qt: pint.Quantity,
        env_height: pint.Quantity,
        env_temperature: pint.Quantity,
        env_qv: pint.Quantity,
        integration_bound_lower: pint.Quantity = None,
        integration_bound_upper: pint.Quantity = None) -> pint.Quantity:
    parcel_density_temperature = density_temperature(parcel_temperature, parcel_qv, parcel_qt)
    env_density_temperature = density_temperature(env_temperature, env_qv, env_qv)

    integrated_positive_buoyancy = 0 * units("J/kg")
    integrated_negative_buoyancy = 0 * units("J/kg")
    lfc = None
    el = None

    env_mse = mpcalc.moist_static_energy(env_height, env_temperature, env_qv)

    # for i in range(len(env_mse)):
    #     print(i, env_height[i], env_temperature[i], env_mse[i])
    
    height_min_mse_idx = np.where(env_mse==np.min(env_mse))[0][0] #FIND THE INDEX OF THE HEIGHT OF MINIMUM MSE
    height_min_mse = env_height[height_min_mse_idx]

    for i in range(len(parcel_height) - 1, 0, -1):
        z0 = parcel_height[i]
        dz = parcel_height[i] - parcel_height[i - 1]

        if integration_bound_lower != None:
            if z0 < integration_bound_lower:
                continue 

        if integration_bound_upper != None:
            if z0 > integration_bound_upper:
                continue

        T_rho_0 = linear_interp(env_height, env_density_temperature, z0)

        T_rho = parcel_density_temperature[i]

        buoyancy = g * (T_rho - T_rho_0) / T_rho_0

        # print(z0, buoyancy)

        ### MARK FIRST POSITIVE BUOYANCY HEIGHT AS EL
        if buoyancy > 0 * g.units and el == None:
            el = z0

        ### IF LFC IS NOT YET REACHED, INTEGRATE ALL POSITIVE BUOYANCY
        if buoyancy > 0 * g.units and lfc == None:
            integrated_positive_buoyancy += buoyancy * dz
        
        # print("z0 < height_min_mse:", z0, height_min_mse)
        ### MARK FIRST NEGATIVE BUOYANCY HEIGHT BELOW MIN_MSE AS LFC
        if z0 < height_min_mse and buoyancy < 0 * g.units:
            integrated_negative_buoyancy += buoyancy * dz

            if lfc == None:
                lfc = z0

    if lfc == None:
        lfc = env_height[0]

    return integrated_positive_buoyancy, integrated_negative_buoyancy, lfc, el

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
    if isinstance(temperature, list):
        if temperature[0] == None:
            len_t = len(temperature)

            return [None] * len_t

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

    q_condensate = qt_parcel - qv_parcel
    ql_parcel = q_condensate * (1 - omega)
    qi_parcel = q_condensate * omega

    c_pm = (1 - qt_parcel) * c_pd + qv_parcel * c_pv + ql_parcel * c_pl + qi_parcel * c_pi
    
    density_temperature_parcel = density_temperature(temperature_parcel, qv_parcel, qt_parcel)
    density_temperature_env = density_temperature(temperature_env, qv_env, qv_env)

    buoyancy = g * (density_temperature_parcel - density_temperature_env)/density_temperature_env

    L_v = L_v_trip + (temperature_parcel - T_trip)*(c_pv - c_pl)
    L_i = L_i_trip + (temperature_parcel - T_trip)*(c_pl - c_pi)

    L_s = L_v + omega * L_i

    Q_vsl = q_vsl/(phi - phi*qt_parcel + qv_parcel)
    Q_vsi = q_vsi/(phi - phi*qt_parcel + qv_parcel)

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

    return -(term_1 + term_2 + term_3 - term_4 - term_5) / (term_6 - term_7 + term_8)