# ecape-parcel-py
A simple Python package that computes ECAPE values and parcel paths.

# Installation
Installation through PyPI is recommended. Copy-paste the following line into your terminal.

`pip install ecape-parcel`

After that, include the following line in your Python script, and you should be good to go.

`from ecape_parcel.calc import calc_ecape_parcel`

# How To Use
This package has been written with the intention of using the same input data that MetPy can handle. This example script was used as a test during development, utilizing <a href="https://github.com/kylejgillett/sounderpy">Kyle Gillett's SounderPy</a> to get test data.

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

parcel_p, parcel_z, parcel_T, parcel_Td = calc_ecape_parcel(p, z, T, Td, u, v, True)
```

The last parameter controls the vertical resolution of the returned parcel path. If set to `False`, the parcel path will have a constant vertical resolution of 20 meters, which is the `dz` value used by the parcel path solver internally. If set to `True`, the parcel path will only contain values from the exact same pressure levels as the pressure array used as an input.

The returned parcel path is a tuple containing one list each for parcel pressure, height, temperature, and dewpoint. This is to allow for virtual temperature to be plotted if the developer wishes.
