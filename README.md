# ecape-parcel-py
A simple Python package that computes ECAPE values and parcel paths.

!! This code is currently going through a rework, which is nearly complete. All parcel ascent configurations now work, but there is a possible bug in setting the entrainment rate that I am currently working out. !!

# Verification

This package uses Equation 24 from [Peters et. al. 2022](https://journals.ametsoc.org/view/journals/atsc/79/3/JAS-D-21-0118.1.xml). I have double-checked the outputs of my code against the outputs of Dr. John Peters's [ECAPE_FUNCTIONS](https://figshare.com/articles/code/ECAPE_scripts/21859818?file=42303630) script for the four different configurations of parcel ascent and plotted the results of both his parcel and mine using Kyle Gillett's [SounderPy](https://github.com/kylejgillett/sounderpy). Accuracy is to within about 1 Kelvin, and I expect that these errors are caused by differences in the numerical integration methods used.

1) Non-Entraining Pseudoadiabatic Ascent (the currently conventional way to compute parcel paths) ![RAP_Wylie-TX_20191021-03_entr-off_pseudo-on](https://github.com/a-urq/ecape-parcel-py/assets/114271919/0ec746b8-61f7-47fa-bcdb-a206600d34f5)
2) Entraining Pseudoadiabatic Ascent ![RAP_Wylie-TX_20191021-03_entr-on_pseudo-on](https://github.com/a-urq/ecape-parcel-py/assets/114271919/4f7b4a4d-c825-48b0-8884-34fb7fbc3f2a)
3) Non-Entraining Irreversible Adiabatic Ascent ![RAP_Wylie-TX_20191021-03](https://github.com/a-urq/ecape-parcel-py/assets/114271919/beb97807-fa8b-4e10-906a-a7ee341eb2cc)
4) Entraining Irreversible Adiabatic Ascent ![RAP_Wylie-TX_20191021-03_entr-on_pseudo-off](https://github.com/a-urq/ecape-parcel-py/assets/114271919/7419e28c-26a0-4547-89eb-902858c4f995)


# Installation
Installation through PyPI is recommended. Be advised that the currently released version uses an incorrect formula, and work on this problem is ongoing. Copy-paste the following line into your terminal.

`pip install ecape-parcel`

After that, include the following line in your Python script, and you should be good to go.

`from ecape_parcel.calc import calc_ecape_parcel`

# How To Use
This package has been written with the intention of using the same input data format as MetPy. This example script was used as a test during development, utilizing <a href="https://github.com/kylejgillett/sounderpy">Kyle Gillett's SounderPy</a> to get test data. This script uses the current pre-release version of ecape-parcel-py, not the release version on PyPI.

```python
import sounderpy as spy
from ecape_parcel.calc import calc_ecape_parcel

# This file uses real-world meteorological data as a test for the ECAPE parcel code. 
# May be removed from repository later on if any circular dependency issues come up
year  = '2013' 
month = '05'
day   = '20'
hour  = '17'
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
```

The last parameter controls the vertical resolution of the returned parcel path. If set to `False`, the parcel path will have a constant vertical resolution of 20 meters, which is the `dz` value used by the parcel path solver internally. If set to `True`, the parcel path will only contain values from the exact same pressure levels as the pressure array used as an input.

The returned parcel path is a tuple containing one list each for parcel pressure, height, temperature, and dewpoint. This is to allow for virtual temperature to be plotted if the developer wishes.
