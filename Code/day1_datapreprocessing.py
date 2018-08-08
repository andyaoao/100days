import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

dataset = pd.read_csv('./Datasets/Data.csv')

X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , -1].values

print (X)

imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:2])
X[ : , 1:2] = imputer.transform(X[ : , 1:2])

print (X)
