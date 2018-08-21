# 6日目データ処理（欠損値処理）　Day 6 Dealing with Missing Value

本日の目標は
1. 欠損値の検知
2. 欠損値の除外
3. 欠損値の置換

## Step 1: データを取り込む
```python
import pandas as pd

#4列のデータセットを読み込む
dataset = pd.read_csv('./Datasets/Data.csv')

# 列全体に対して、欠損しているかどうか
print(dataset.isnull().all())

# 列全体に対して、欠損値が存在するかないか
print(dataset.isnull().any())

# 列ごとの欠損値個数合計
print(dataset.isnull().sum())
```
## Step 2: 欠損値の除外
```python
# 列全体が欠損値の場合、除外
print(dataset.dropna(how='all'))
# 行全体が欠損値の場合、除外
print(dataset.dropna(how='all', axis=1))

# 欠損値が一つでも含まれる行が削除される(二つ同意)
print(dataset.dropna(how='any'))
print(dataset.dropna())

# 個数(下記は3)を指定すると含まれる欠損値の数に応じて行を削除
print(dataset.dropna(thresh=3))
```
## Step 3: 欠損値の置換
```python
# sklearn imputer で欠損値を置換
from sklearn.preprocessing import Imputer
X = dataset
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:2])
X[ : , 1:2] = imputer.transform(X[ : , 1:2])

# pandas　で欠損値を置換
# 平均値で置き換える
print(dataset.fillna(df.mean()))
# 前の値で置き換える
print(df.fillna(method='ffill'))
# 後の値で置き換える
print(df.fillna(method='bfill'))

```

### 参考資料
欠損値処理　https://note.nkmk.me/python-pandas-nan-dropna-fillna/
