import sounderpy as spy
from ecape_parcel import ecape_parcel

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

ecape_parcel(p, z, T, Td, u, v)

spy.metpy_sounding(clean_data)
