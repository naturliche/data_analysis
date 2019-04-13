# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:29:46 2019

@author: natur
"""

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

file1 = r'C:\Users\natur\Downloads\IPYNB\DS_80_SEC\DS\hurs_Amon_CESM1-BGC_esmHistorical_r1i1p1_185001-200512.nc'
file2 = r'C:\Users\natur\Downloads\IPYNB\DS_80_SEC\DS\tas_Amon_CESM1-BGC_esmHistorical_r1i1p1_185001-200512.nc'

ds1 = Dataset(file1,'r')
ds2 = Dataset(file2,'r')

hurs = ds1.variables['hurs'][:,:,:]
tas = ds2.variables['tas'][:,:,:]
lat = ds1.variables['lat'][:]
lon = ds1.variables['lon'][:]

#随机
lon_index = int(101.2 / (360/len(lon)))
lat_index = int((28.5+90) / (360/len(lat)))

hurs_data = hurs[:,lat_index,lon_index]
tas_data = tas[:,lat_index,lon_index]

X = np.hstack((hurs_data[90:-2].reshape(-1,1),tas_data[90:-2].reshape(-1,1)))
y = np.full([len(tas_data)-92,1],np.nan)
for i in range(len(y)):
    y[i] = (np.mean(tas_data[90+i:90+i+3])-np.mean(tas_data[i:i+90])) > 0
    
    
X_train = []
y_train = []
for i in range(10,int(len(X)*0.8)):
    X_train.append(X[i-10:i,:])
    y_train.append(y[i])
X_train,y_train = np.array(X_train),np.array(y_train)  
#数据类型转换 

    
X_test = []
y_test = []
for i in range(int(len(X)*0.8),len(X)):
    X_test.append(X[i-10:i,:])
    y_test.append(y[i])
X_test,y_test = np.array(X_test),np.array(y_test)   

#模型
mymodel =Sequential()
mymodel.add(LSTM(10,input_shape = (X_train.shape[1],X_train.shape[2])))
mymodel.add(Dense(units=1,activation='sigmoid'))
mymodel.compile(optimizer='adam',loss='mse')
mymodel.fit(X_train,y_train)

y_pre = mymodel.predict(X_test)

#post process
#y_test 和 y_pre比较








