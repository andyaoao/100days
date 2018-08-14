# 5日目ロジスティック回帰分析　Day 5 Logistics Linear Regression 1

本日の目標は
1. データを取り込む
2. multiple linear regressionのモデルを作成
3. モデルで予測結果を作成

## Step 1: データを取り込む
```python
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

#5列のデータセットを読み込む
dataset = pd.read_csv('./Datasets/Social_Network_Ads.csv')
# 3, 4列をXとして格納
X = dataset.iloc[:, [2, 3]].values
# 最後の列はY(1行)として格納
y = dataset.iloc[:, 4].values

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 詳細のデータ前処理はday1にご参考
```
## Step 2: Logistics linear regressionのモデルを作成
```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
```
## Step 3: モデルで予測結果を作成
```python
y_pred = classifier.predict(X_test)

```

## 補足
### ロジスティック回帰と線形回帰の違い
ロジスティック回帰のアウトプットは0か1(discrete);線形回帰のアウトプットは数値(continuous)。

### 参考資料
ロジスティック回帰　https://qiita.com/yshi12/items/3dbd336bd9ff7a426ce9
