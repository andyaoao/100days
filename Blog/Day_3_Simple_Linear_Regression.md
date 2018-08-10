# 3日目単回帰分析　Day 3 Simple Linear Regression

本日の目標は
1. データを取り込む
2. simple linear regressionのモデルを作成
3. モデルで予測結果を作成
4. 図で出力

## Step 1: データを取り込む
```python
import pandas as pd
import numpy as np

# 2列のデータセットを読み込む
dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values

# 75-25でtraining setとtest setを分割
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.25, random_state = 0)
```

## Step 2: simple linear regressionのモデルを作成
```python
# training setでモデル作成
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)
```

## Step 3: モデルで予測結果を作成
```python
# Step 2で作成したモデルをベースで予測
Y_pred = regressor.predict(X_test)
```
## Step 4: 図で出力
```python
import matplotlib.pyplot as plt

# training setの散布図をHydrogenで出力
plt.scatter(X_train , Y_train, color = 'red')
plt.plot(X_train , regressor.predict(X_train), color ='blue')
plt.show()

# 予測の散布図をHydrogenで出力
plt.scatter(X_test , Y_test, color = 'red')
plt.plot(X_test , regressor.predict(X_test), color ='blue')
plt.show()
```

## 補足
### モデルの精度を検証すべき
MSEとRMSEでモデルの精度を評価できる。
>MSE:metrics.mean_squared_error(y_train, y_train_pred)
>RMSE:root(metrics.mean_squared_error(y_train, y_train_pred))

### Feature Scaling
regressorはScalingの機能を持っているため、処理しなくて済む。

### 参考資料
using jupyter in ATOM http://hogeai.hatenablog.com/entry/2018/01/20/044158
several KPI to exam model https://mathwords.net/rmsemae
