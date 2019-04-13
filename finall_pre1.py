# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:58:51 2019

@author: natur
"""

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import csv

f1 = Dataset(r"C:\Users\natur\Downloads\IPYNB\DS_80_SEC\DS\hurs_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012.nc")
f2 = Dataset(r"C:\Users\natur\Downloads\IPYNB\DS_80_SEC\DS\evspsbl_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012.nc")
f3 = Dataset(r"C:\Users\natur\Downloads\IPYNB\DS_80_SEC\DS\tas_Amon_CESM1-BGC_rcp85_r1i1p1_200601-210012.nc")
#f4 = Dataset(r"C:\Users\natur\Downloads\IPYNB\DS_80_SEC\DS\lai_Lmon_MIROC-ESM-CHEM_rcp85_r1i1p1_200601-210012.nc")
f5 = Dataset(r"C:\Users\natur\Downloads\IPYNB\DS_80_SEC\DS\treeFrac_Lmon_CMCC-CESM_rcp85_r1i1p1_200601-210012.nc")

hurs = f1.variables['hurs'][:,:,:]
evspsbl = f2.variables['evspsbl'][:,:,:]
tas = f3.variables['tas'][:,:,:]
#lai = f4.variables['lai'][:,:,:]
treeFrac = f5.variables['treeFrac'][:,:,:]
i = f5.variables['i'][:]
j = f5.variables['j'][:]
lat = f1.variables['lat'][:]
lon = f1.variables['lon'][:]
lat1 = f5.variables['lat'][:,0]
lon1 = f5.variables['lon'][0,:]
#lat2 = f4.variables['lat'][:]
#lon2 = f4.variables['lon'][:]
time = f1.variables['time'][:]


#凉山 28.5°N，101.2°E
lon_index = int(101.2 / (360/len(lon)))
lat_index = int((28.5+90) / (360/len(lat)))
lon1_index = int(101.2 / (360/len(lon1)))
lat1_index = int((90-28.5) / (360/len(lat1)))
#lon2_index = int(101.2 / (360/len(lon2)))
#lat2_index = int((28.5+90) / (360/len(lat2)))
hurs_ny = hurs[:,lat_index,lon_index]
evspsbl_ny = evspsbl[:,lat_index,lon_index]
tas_ny = tas[:,lat_index,lon_index]
treeFrac_ny = treeFrac[:,lat1_index,lon1_index]
#lai_ny = lai[:,lat2_index,lon2_index]

x = hurs_ny
y = evspsbl_ny
z = tas_ny
k = treeFrac_ny


with open("fina_pre.csv","w") as f:
    writer = csv.writer(f,delimiter="|")
    header = ["hurs","evspsbl","tas","treeFrac"]
    writer.writerow(header)
    writer.writerows(zip(x,y,z,k))
    
f.close()


