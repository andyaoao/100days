import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#day 3 start

# 2列のデータセットを読み込む
dataset = pd.read_csv('./Datasets/studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

# 75-25でtraining setとtest setを分割
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.25, random_state = 0)

# LinearRegressionのモデルを作成し、training setを投入
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# test setを通して、予測を作成。
Y_pred = regressor.predict(X_test)

# training setの散布図
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.show()

# 予測の散布図
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.show()

# day 4 追加
# MSEの計算
print (mean_squared_error(Y_test, Y_pred))
