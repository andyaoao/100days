# 13日目Grid Searchでパフォーマンスチューニング　Day 13 Grid Search

本日の目標は
1. データを取り込みから整理まで
2. Naive Bayes Classifierのロジック理解
3. Naive Bayes Classifierモデル作成と予測結果を作成

## Step 1: データを取り込みから整理まで
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#5列のデータセットを読み込む
dataset = pd.read_csv('./Datasets/Social_Network_Ads.csv')
# 3, 4列をXとして格納
X = dataset.iloc[:, [2, 3]].values
# 最後の列はY(1行)として格納
y = dataset.iloc[:, 4].values

# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

```
## Step 2: Naive Bayes Classifierのロジック理解
特定のイベントが発生する前提で、あるイベント発生する可能性(Probability)
P(A|B) is “Probability of A given B”, the probability of A given that B happens
P(A) is Probability of A
P(B|A) is “Probability of B given A”, the probability of B given that A happens
P(B) is Probability of B


```python
from sklearn.naive_bayes import GaussianNB

# training setをNaive bayesのモデルを訓練する
model = GaussianNB()
model.fit(X_train, y_train)

#Predict Output
y_pred = model.predict(y_test)
print y_pred
```


## 補足

### 参考資料
Kernel Trickの説明 https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-kernel-%E5%87%BD%E6%95%B8-47c94095171  
Kernel Trickの説明 https://chtseng.wordpress.com/2017/02/04/support-vector-machines-%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%A9%9F/  
Grid Searchの実現 https://qiita.com/arata-honda/items/8d08f31aa7d7cbae4c91  
Grid Searchの実現 https://blog.amedama.jp/entry/2017/09/05/221037  
