# ecape-parcel-py
A simple Python package that computes ECAPE values and parcel paths. Code has been verified by checking it against the ECAPE_FUNCTIONS script written by Dr. John Peters.

# Installation
Installation through PyPI is recommended. Be advised that the currently released version uses an incorrect formula, and work on this problem is ongoing. Copy-paste the following line into your terminal.

`pip install ecape-parcel`

After that, include the following line in your Python script, and you should be good to go.

`from ecape_parcel.calc import calc_ecape_parcel, density_temperature`

# How To Use
This package has been written with the intention of using the same input data format as MetPy. This example script was used as a test during development, utilizing <a href="https://github.com/kylejgillett/sounderpy">Kyle Gillett's SounderPy</a> to get test data.

```python
import sounderpy as spy
from ecape_parcel.calc import calc_ecape_parcel, density_temperature

# This file uses real-world meteorological data as a test for the ECAPE parcel code. 
# May be removed from repository later on if any circular dependency issues come up
year  = '2023' 
month = '05'
day   = '12'
hour  = '00'
latlon = [35.18, -97.44]
method = 'rap' 

raw_data = spy.get_model_data(method, latlon, year, month, day, hour)

clean_data = spy.parse_data(raw_data)

p = clean_data['p']
T = clean_data['T']
Td = clean_data['Td']
z = clean_data['z']
u = clean_data['u']
v = clean_data['v'] 

parcel_p, parcel_z, parcel_T, parcel_qv, parcel_qt = calc_ecape_parcel(p, z, T, Td, u, v, True)

# The last parameter controls the vertical resolution of the returned parcel path.
# If set to `False`, the parcel path will have a constant vertical resolution of 20
# meters, which is the `dz` value used by the parcel path solver internally. If set
# to `True`, the parcel path will only contain values from the exact same pressure
# levels as the pressure array used as an input.

parcel_T_rho = density_temperature(parcel_T, parcel_qv, parcel_qt)
```

Plot `parcel_T_rho` on a Skew-T and you'll be good to go. I've also plotted an undiluted CAPE parcel alongside the ECAPE parcel to better illustrate the difference.

Note: While you could plot parcel_T on its own, using the density temperature is required to get the most amount of benefit from Dr. Peters's work.

![RAP_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa](https://github.com/a-urq/ecape-parcel-py/assets/114271919/90b381ff-7cfa-47c6-84b6-cce2739f8dbb)

This parcel trace may look a little different than you were expecting. As part of Dr. Peters's work with deriving lapse rate formulas that considered entrainment, he also accounted for the effect of cloud condensate on the density of the parcel, and therefore the buoyancy. When accounting for this, the parcel is said to be undergoing "irreversible adiabatic ascent". I usually call it "true adiabatic ascent" since I usually don't feel like explaining what's so irreversible about it.

 This causes two noticeable effects from what we're used to.

* Due to the weight of cloud condensate, the buoyancy in the lower levels is significantly reduced, up to even 30%-40%.
* Since the liquid cloud droplets are still in the parcel, they can freeze and release latent heat. This release of latent heat causes a significant *increase* in buoyancy in the upper levels and also raises the EL by a significant amount.

![RAP_Wylie-TX_20191021-03_entr-off_pseudo-off](https://github.com/a-urq/ecape-parcel-py/assets/114271919/51bd953e-72e4-4ae4-bbc8-5e32154ee67c)

In cases where you would prefer to use the traditional method of parcel ascent that assumes all precipitate is removed from the parcel immediately, called "pseudoadiabatic ascent", ecape-parcel-py supports that too. The following code shows all four parcel ascent processes available to users.

```python
# Uses Non-Entraining Pseudoadiabatic Ascent
# Currently the traditional method of computing parcels
# Returns a slightly different parcel from MetPy since it uses a more detailed lapse rate equation
calc_ecape_parcel(p, z, T, Td, u, v, True, entrainment_switch=False, pseudoadiabatic_switch=True)

# Uses Entraining Pseudoadiabatic Ascent
calc_ecape_parcel(p, z, T, Td, u, v, True, entrainment_switch=True, pseudoadiabatic_switch=True)

# Uses Non-Entraining True Adiabatic Ascent
calc_ecape_parcel(p, z, T, Td, u, v, True, entrainment_switch=False, pseudoadiabatic_switch=False)

# Uses Entraining True Adiabatic Ascent
calc_ecape_parcel(p, z, T, Td, u, v, True, entrainment_switch=True, pseudoadiabatic_switch=False)
```

Plotting pseudoadiabatic parcels for undiluted CAPE and ECAPE might return a result more like what you're expecting.

![aaaaaaaaaa](https://github.com/a-urq/ecape-parcel-py/assets/114271919/fe7ac6ce-cc00-47e0-9455-073be543bf7d)

Also keep in mind that entrainment rates can vary widely depending on many factors, including EL height and storm-relative inflow magnitude. Most soundings here are of supercells with very low entrainment rates. Most non-supercell environments will have much narrower updrafts and therefore much higher entrainment rates.

![aaaaaaaaaa-nonsup](https://github.com/a-urq/ecape-parcel-py/assets/114271919/32bdfdf6-b2d2-4f1a-b7d1-3ede8199e85b)

Additional configuration options are available for users who want to make more specific calculations.

```python

# Computes a parcel for Surface Based ECAPE
calc_ecape_parcel(p, z, T, Td, u, v, True, entrainment_switch=True, pseudoadiabatic_switch=False, cape_type="surface_based")

# Computes a parcel for 100 mb Mixed Layer ECAPE
calc_ecape_parcel(p, z, T, Td, u, v, True, entrainment_switch=True, pseudoadiabatic_switch=False, cape_type="mixed_layer")

# Computes a parcel for Most Unstable ECAPE (default)
calc_ecape_parcel(p, z, T, Td, u, v, True, entrainment_switch=True, pseudoadiabatic_switch=False, cape_type="most_unstable")

# Uses the Left-Moving 0-500 m storm relative inflow
calc_ecape_parcel(p, z, T, Td, u, v, True, entrainment_switch=True, pseudoadiabatic_switch=False, storm_motion_type="left_moving", inflow_layer_bottom: pint.Quantity = 0 * units.kilometer, inflow_layer_top: pint.Quantity = 0.5 * units.kilometer)

# Uses the Mean Wind 0-3 km storm relative inflow
calc_ecape_parcel(p, z, T, Td, u, v, True, entrainment_switch=True, pseudoadiabatic_switch=False, storm_motion_type="mean_wind", inflow_layer_bottom: pint.Quantity = 0 * units.kilometer, inflow_layer_top: pint.Quantity = 3 * units.kilometer)

# Uses user-computed CAPE, LFC, and EL values
calc_ecape_parcel(p, z, T, Td, u, v, True, entrainment_switch=True, cape=3500 * units("J/kg"), lfc=500 * units("m"), cape=12500 * units("m"))
```

# Verification

This package uses Equation 24 from [Peters et. al. 2022](https://journals.ametsoc.org/view/journals/atsc/79/3/JAS-D-21-0118.1.xml). I have double-checked the outputs of my code against the outputs of Dr. Peters's [ECAPE_FUNCTIONS](https://figshare.com/articles/code/ECAPE_scripts/21859818?file=42303630) script for the four different configurations of parcel ascent and plotted the results of both his parcel and mine using Kyle Gillett's [SounderPy](https://github.com/kylejgillett/sounderpy). Accuracy is to within about 0.75 Kelvins, and I expect that these discrepancies are caused by differences in the numerical integration methods used.

1) Non-Entraining Pseudoadiabatic Ascent (the currently conventional way to compute parcel paths) ![RAP_Wylie-TX_20191021-03_entr-off_pseudo-on](https://github.com/a-urq/ecape-parcel-py/assets/114271919/ee2a2521-b7f9-4776-8b3c-a2f0456f58cb)
2) Entraining Pseudoadiabatic Ascent![RAP_Wylie-TX_20191021-03_entr-on_pseudo-on](https://github.com/a-urq/ecape-parcel-py/assets/114271919/b3f08524-15a9-4a82-b429-77f3c5a79180)
3) Non-Entraining Irreversible Adiabatic Ascent![RAP_Wylie-TX_20191021-03_entr-off_pseudo-off](https://github.com/a-urq/ecape-parcel-py/assets/114271919/8e508bca-426d-45c1-8e5b-f641b953f6f1)
4) Entraining Irreversible Adiabatic Ascent![RAP_Wylie-TX_20191021-03_entr-on_pseudo-off](https://github.com/a-urq/ecape-parcel-py/assets/114271919/3176439f-7945-4265-aed1-5d6138c03772)

