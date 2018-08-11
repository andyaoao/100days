import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#day 4 start

#5列のデータセットを読み込む
dataset = pd.read_csv('./Datasets/50_Startups.csv')
# 最後の列以外はXとして格納
X = dataset.iloc[ : , :-1].values
# 最後の列はY(1行)として格納
Y = dataset.iloc[ : ,  4 ].values

# Xの第3列はカテゴリタイプデータ、コード化する。
labelencoder = LabelEncoder()
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# ラベルCityは高度相関のため、n-1個のみ採用
X = X[: , 1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)


print (y_pred)
