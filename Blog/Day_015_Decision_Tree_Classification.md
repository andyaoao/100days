# 14日目D決定木分類　Day 13 Decision Tree Classification

本日の目標は
1. データを取り込みから整理まで
2. Decision Treeのロジック理解
3. Decision Treeモデル作成と予測結果を作成

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
## Step 2: Decision Treeのロジック理解
ロジック：データ全体に対して、複数の条件で次々2分類する。  
パラメータ：
criterion：Giniは連続データに向いている一方、Entropyは分類に向いているようだ。
> Gini is intended for continuous attributes, and Entropy for attributes that occur in classes

max_depth：ツリーの最大レベル(過学習を防ぐ)
min_samples_split：分類後のセットはサンプル何個がある(小さければ小さいほど、過学習、パフォマンスが遅いの恐れがある)

DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
max_features=None, random_state=None, max_leaf_nodes=None,
min_impurity_split=1e-07, class_weight=None, presort=False)

## Step 3: Decision Treeモデル作成と予測結果を作成
```python
from sklearn.tree import DecisionTreeClassifier

# DecisionTreeを実装(今回は分類を求めたいため、criterionはentropyとする)
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Predict Output
y_pred = model.predict(X_test)
print y_pred
```

## Step 4: Decision Treeの結果を図で確認

```python
# Training set を図で表現する
# day 11で作った等高線のfunction
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plot_decision_regions(X_train, y_train, classifier=classifier)
plt.title('DecisionTree (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Test set を図で表現する
plot_decision_regions(X_test, y_test, classifier=classifier)
plt.title('DecisionTree (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

## 補足

### 参考資料
Decision Treeの実装 https://qiita.com/kibinag0/items/6e06561aceeb27ea86d8  
Decision Trees: “Gini” vs. “Entropy” criteria	https://www.garysieling.com/blog/sklearn-gini-vs-entropy-criteria8-47c94095171  
