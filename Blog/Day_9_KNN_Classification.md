# 9日目KNN分類　Day 8 KNN Classification

本日の目標は
1. データを取り込みから整理まで
2. multiple linear regressionのモデルを作成
3. モデルで予測結果を作成

## Step 1: データを取り込みから整理まで
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# データを読み込む
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# datasetを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

```
## Step 2: Logistics linear regressionのモデルを作成
```python
from sklearn.neighbors import KNeighborsClassifier

# minkowskiはsum(|x - y|^p)^(1/p)
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

```
## Step 3: モデルで予測結果を作成
```python
# 予測結果
y_pred = classifier.predict(X_test)

```

## 補足

### 参考資料
KNN metricの分類　https://ameblo.jp/cognitive-solution/entry-12289785974.html  
どの方法で距離を計算したほうがいい（図）　https://qiita.com/yhyhyhjp/items/84c1b4acebb018d09b9c　　
