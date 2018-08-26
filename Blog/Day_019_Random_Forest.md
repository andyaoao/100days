# 19日目Random Forest　Day 19 Random Forest

本日の目標は
1. データを取り込みから整理まで
2. Random Forestのロジック理解
3. Random Forestモデル作成と予測結果を作成

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

# DecisionTreeには、feature scalingを実施する必要がある
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

```
## Step 2: Random Forestのロジック理解
複数のDecision Treeを作成する。
ランダムで切られたサンプルをそれぞれDecision Treeモデルを構成する。
inputが入ったら、全てのモデルがそれぞれの答えを持って、多数決で予測結果を決める(回帰の場合は平均)

## Step 3: Random Forestモデル作成と予測結果を作成
```python
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

```

## Step 4: Random Forestの結果を図で確認

```python
# Training set を図で表現する
# day 11で作った等高線のfunction
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plot_decision_regions(X_train, y_train, classifier=classifier)
plt.title('Random Forest Classification (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Test set を図で表現する
plot_decision_regions(X_test, y_test, classifier=classifier)
plt.title('Random Forest Classification (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

## 補足

### 参考資料
Random Forestの実装 https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-5%E8%AC%9B-%E6%B1%BA%E7%AD%96%E6%A8%B9-decision-tree-%E4%BB%A5%E5%8F%8A%E9%9A%A8%E6%A9%9F%E6%A3%AE%E6%9E%97-random-forest-%E4%BB%8B%E7%B4%B9-7079b0ddfbda  
Decision TreeとRandom forest	https://qiita.com/AwaJ/items/3c02ff64b6a89e1a96f1  
