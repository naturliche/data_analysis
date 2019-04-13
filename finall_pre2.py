# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 21:38:37 2019

@author: natur
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

dataset = pd.read_csv(r'C:\Users\natur\Downloads\IPYNB\DS_80_SEC\HELLO\fina_pre.csv')
x = dataset.iloc[:,:].values

'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = 1.0e+20, strategy = 'mean')
imputer = imputer.fit(x[:,:])
'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 1.0e+20, strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:,:])
x[:,:] = imputer.transform(x[:,:])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
x1 = scaler.transform(x)

from scipy import stats
mydata1 = dataset['hurs']
mydata_t1,lamda1 = stats.boxcox(mydata1)
mydata2 = dataset['evspsbl']
mydata_t2,lamda2 = stats.boxcox(mydata2)
mydata3 = dataset['tas']
mydata_t3,lamda3 = stats.boxcox(mydata3)
mydata4 = dataset['treeFrac']
mydata_t4,lamda4 = stats.boxcox(mydata4)


X = np.hstack((mydata1[100:-2].values.reshape(-1,1),mydata2[100:-2].values.reshape(-1,1),mydata3[100:-2].values.reshape(-1,1),mydata4[100:-2].values.reshape(-1,1)))
y1 = np.full([len(mydata1)-102,1],np.nan)
for i in range(len(y1)):
    y1[i] = (np.mean(mydata1[100+i:100+i+3])-mydata1[173:174]) > 0
y2 = np.full([len(mydata2)-102,1],np.nan)
for m in range(len(y2)):
    y2[i] = (np.mean(mydata2[100+m:100+m+3])-mydata2[173:174]) > 0

    
X_train = []
y1_train = []
for j in range(10,int(len(X)*0.8)):
    X_train.append(X[j-10:j,:])
    y1_train.append(y1[j])
X_train,y1_train = np.array(X_train),np.array(y1_train)  
y2_train = []
for n in range(10,int(len(X)*0.8)):
    y2_train.append(y2[n])
y2_train = np.array(y2_train)  

    
X_test = []
y1_test = []
for k in range(int(len(X)*0.8),len(X)):
    X_test.append(X[k-10:k,:])
    y1_test.append(y1[k])
X_test,y1_test = np.array(X_test),np.array(y1_test) 
y2_test = []
for a in range(int(len(X)*0.8),len(X)):
    y2_test.append(y2[a])
y2_test = np.array(y2_test)   

mymodel = Sequential()
mymodel.add(LSTM(13,input_shape = (X_train.shape[1],X_train.shape[2])))
mymodel.add(Dense(units=1,activation='sigmoid'))
mymodel.compile(optimizer='adam',loss='mse')
mymode2 = Sequential()
mymode2.add(LSTM(13,input_shape = (X_train.shape[1],X_train.shape[2])))
mymode2.add(Dense(units=1,activation='sigmoid'))
mymode2.compile(optimizer='adam',loss='mse')
mymodel.fit(X_train,y1_train)
mymode2.fit(X_train,y2_train)

y1_pre = mymodel.predict(X_test)
y2_pre = mymode2.predict(X_test)

from sklearn import metrics
print('MAE',metrics.mean_absolute_error(y1_test,y1_pre))
print('MSE',metrics.mean_squared_error(y1_test,y1_pre))
print('RMSE',np.sqrt(metrics.mean_squared_error(y1_test,y1_pre)))

from sklearn import metrics
print('MAE',metrics.mean_absolute_error(y2_test,y2_pre))
print('MSE',metrics.mean_squared_error(y2_test,y2_pre))
print('RMSE',np.sqrt(metrics.mean_squared_error(y2_test,y2_pre)))





