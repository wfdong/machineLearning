# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np #math lib
import matplotlib.pyplot as plt #visualized lib for numpy
import pandas as pd #data analysis toolkit

#importing the dataset
dataset = pd.read_csv('Data.csv');
X = dataset.iloc[:, :-1].values #[:, :-1], first : means all lines, second :-1 means all column except last 1 column
Y = dataset.iloc[:, 3].values

# Taking care of missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[:,1:3])  # all lines, columns from index 1 to 2
#X[:, 1:3] = imputer.transform(X[:, 1:3])

from sklearn.impute import SimpleImputer
simpleImputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
simpleImputer = simpleImputer.fit(X[:,1:3])
X[:, 1:3] = simpleImputer.transform(X[:, 1:3])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

#dummy categorical
ct = ColumnTransformer([('encoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Euclidean distance...
# Feature scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)