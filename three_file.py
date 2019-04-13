# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 16:01:23 2019

@author: natur
"""

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

f1 = Dataset(r"C:\Users\natur\Downloads\IPYNB\DS_80_SEC\DS\hurs_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012.nc")
f2 = Dataset(r"C:\Users\natur\Downloads\IPYNB\DS_80_SEC\DS\mrsos_Lmon_CESM1-BGC_rcp85_r1i1p1_200601-210012.nc")
f3 = Dataset(r"C:\Users\natur\Downloads\IPYNB\DS_80_SEC\DS\evspsbl_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012.nc")

hurs = f1.variables['hurs'][:,:,:]
mrsos = f2.variables['mrsos'][:,:,:]
evspsbl = f3.variables['evspsbl'][:,:,:]
lat = f1.variables['lat'][:]
lon = f1.variables['lon'][:]
time = f1.variables['time'][:]


#凉山 28.5°N，101.2°E
lon_index = int(101.2 / (360/len(lon)))
lat_index = int((28.5+90) / (360/len(lat)))
hurs_ny = hurs[:,lat_index,lon_index]
mrsos_ny = mrsos[:,lat_index,lon_index]
evspsbl_ny = evspsbl[:,lat_index,lon_index]

x = time
y = hurs_ny
z = mrsos_ny
k = evspsbl_ny


with open("liner2.csv","w") as f:
    writer = csv.writer(f,delimiter="|")
    header = ["time","hurs","mrsos","evspsbl"]
    writer.writerow(header)
    writer.writerows(zip(x,y,z,k))
    
f.close()