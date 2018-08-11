# 3日目単回帰分析　Day 3 Simple Linear Regression

本日の目標は
1. データを取り込む
2. カテゴリデータの処理
3. multiple linear regressionのモデルを作成
4. モデルで予測結果を作成
4. 図で出力

## Step 1: データを取り込む
```python
import pandas as pd
import numpy as np

#5列のデータセットを読み込む
dataset = pd.read_csv('50_Startups.csv')
# 最後の列以外はXとして格納
X = dataset.iloc[ : , :-1].values
# 最後の列はY(1行)として格納
Y = dataset.iloc[ : ,  4 ].values
```

## Step 2: カテゴリデータの処理
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
# Xの第3列はカテゴリタイプデータ、コード化する。
X[: , 3] = labelencoder.fit_transform(X[ : , 3])
# カテゴリタイプデータをダミー変数に変換
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# ダミー変数トラップを避ける処理、ラベルCityは高度相関のため、n-1個のみ採用
X = X[: , 1:]

# データ分割
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
```

## Step 3: multiple linear regressionのモデルを作成
```python
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
```
## Step 4: モデルで予測結果を作成
```python
y_pred = regressor.predict(X_test)

```

## 補足
### ダミー変数の処理
カテゴリデータはダミー変数に変換した後、説明変数の数がカテゴリの数になるため、変数間は関連性を持つことになる。なので、関連性を解消しないといけない。
解消の方法は、カテゴリ変数により生成されたダミー変数の一つを説明変数から排除する。

### 参考資料
dummy variable trap https://analyticstraining.com/understanding-dummy-variable-traps-regression/
