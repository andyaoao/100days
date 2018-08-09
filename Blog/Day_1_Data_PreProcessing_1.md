# １日目データ事前処理-1　Day1 Data PreProcessing - 1

本日の目標
1. libraryのインポート（別途libraryインストールも書いた方がいいかな）
2. csvファイルのインポート
3. 数値の欠損値の処理

## Step 1: libraryのインポート
```Python
import numpy as np
import pandas as pd
```
## Step 2: csvファイルのインポート
```python
# DatasetsフォルダにあるData.csvを読み込む
dataset = pd.read_csv('./Datasets/Data.csv')

# 最初の列から後ろから-1番目の列までXとする
X = dataset.iloc[ : , :-1].values

# -1番目の列をYとする
Y = dataset.iloc[ : , -1].values
```
## Step 3: 数値の欠損値の処理
```python
# 欠損値処理のlibraryをインポート
from sklearn.preprocessing import Imputer

# 欠損値処理方針の定義
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

# 欠損値処理方針をDataFrameに適用
# Xの2-4列目に対して処理方針を適用
imputer = imputer.fit(X[ : , 1:3])

# 処理後の内容を本DataFrameに置換
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
```
## 補足　
### ilocの使い方
DafaFrameより、特定の行、列を指定。
df.iloc[行目, 列目]
```Python
# Dataframeの2と3行目、1と2列目(計4個)
df.iloc[[2,3], [1,2]]
# Dataframeの行全体、最初から-1列目まで
df.iloc[ : , :-1]
```

### pushの時の問題
同じ端末で複数のgit userが入っています。
https://teratail.com/questions/73843

### 参考資料
https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/Code/Day%201_Data%20PreProcessing.md
http://ailaby.com/lox_iloc_ix/
https://qiita.com/kibinag0/items/a940bb53b91757f132cc
