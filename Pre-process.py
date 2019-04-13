import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## understand data
# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# features
# samples
# target
# question



## imputation missing value
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])



## feature scaling
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# standardarization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X1_=scaler.transform(X)

# normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
X2_ = scaler.fit_transform(X)

## feature transform
from scipy import stats
inputdata = pd.read_csv('segmentation.csv')

plt.subplot(2,2,1)
plt.hist(inputdata['AreaCh1'])
plt.subplot(2,2,2)
plt.hist(np.log(inputdata['AreaCh1']+1))
plt.subplot(2,2,3)
plt.hist(inputdata['AreaCh1']**2)
plt.subplot(2,2,4)
plt.hist(1/inputdata['AreaCh1'])

mydata = inputdata['AreaCh1']
np.sum((mydata - np.mean(mydata))**3) / (mydata.size - 1)/ (np.sum(np.sum((mydata - np.mean(mydata))**2) /(mydata.size - 1)))**1.5

# box-cox transform 
# mydata_t_ = (mydata**lamda - 1)/lamda
mydata_t,lamda = stats.boxcox(mydata)
np.sum((mydata_t - np.mean(mydata_t))**3) / (mydata_t.size - 1)/ (np.sum(np.sum((mydata_t - np.mean(mydata_t))**2) /(mydata_t.size - 1)))**1.5


## feature enginerring
import seaborn as sns
corr = inputdata.iloc[:,4:15].corr()
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(round(corr,2), vmin=-1.0, vmax=1.0, annot=True, fmt='.2f',
            cmap="coolwarm", linewidths=.05, linecolor = 'white', 
            cbar=True, cbar_kws={"orientation": "vertical"}, square=False)


## data split
# random sampling
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# k-fold
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
for train, test in kf.split(X):
    print("%s %s" % (train, test))

