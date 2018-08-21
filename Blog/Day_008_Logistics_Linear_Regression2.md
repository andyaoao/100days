# 8日目ロジスティック回帰分析　Day 8 Logistics Linear Regression 2

本日の目標は
1. データを取り込みから整理まで
2. multiple linear regressionのモデルを作成
3. モデルで予測結果を作成

## Step 1: データを取り込みから整理まで
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#iris-dataを読み込む
df = pd.read_csv('./Datasets/iris-data.csv')

# 欠損値検知と修正
df.info()

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# column petal_width_cmに欠損値が存在するレコードを削除
df = df.dropna(subset=['petal_width_cm'])

# 変数間の関係性を図で出力する
sns.pairplot(df, hue='class', size=2.5)
# plt.show()

# classの名称が間違っていることを発見。修正する
df['class'].replace(["Iris-setossa","versicolor"], ["Iris-setosa","Iris-versicolor"], inplace=True)

# logistics regressionのoutputは2択なので、一旦2択のみ残す
final_df = df[df['class'] != 'Iris-virginica']

# outlier存在するかどうかを図でチェック
sns.pairplot(final_df, hue='class', size=2.5)
# Iris-setosaのsepal_width_cmにoutlierが存在する
# Iris-versicolorのsepal_height_cmにoutlierが存在する

# 簡単処理のため、outlierが存在するデータを一旦除外する・
final_df = final_df.drop(final_df[(final_df['class'] == "Iris-setosa") & (final_df['sepal_width_cm'] < 2.5)].index)
final_df = final_df.drop(final_df[(final_df['class'] == "Iris-versicolor") & (final_df['sepal_length_cm'] < 1)].index)

sns.pairplot(final_df, hue='class', size=2.5)
# plt.show()

# outputに対し、コード化する
final_df['class'].replace(["Iris-setosa","Iris-versicolor"], [1,0], inplace=True)
```
## Step 2: Logistics linear regressionのモデルを作成
```python
from sklearn.linear_model import LogisticRegression
# inputとoutputのdfを作成
inp_df = final_df.drop(final_df.columns[[4]], axis=1)
out_df = final_df.drop(final_df.columns[[0,1,2,3]], axis=1)

#　feature scaling
scaler = StandardScaler()
inp_df = scaler.fit_transform(inp_df)

#　data set分割
X_train, X_test, y_train, y_test = train_test_split(inp_df, out_df, test_size=0.2, random_state=42)

# Yをarray化
y_tr_arr = y_train.as_matrix()
y_ts_arr = y_test.as_matrix()

# LogisticRegressionモデルを作成
clf = LogisticRegression()
clf.fit(X_train, y_tr_arr)

```
## Step 3: モデルで予測結果を作成
```python
# 予測結果
pred  =  clf.predict(X_test)
print (pred)

```

## 補足

### 参考資料
ロジスティック回帰とcost functionの比較　https://github.com/SSaishruthi/LogisticRegression_Vectorized_Implementation/blob/master/Logistic_Regression.ipynb
