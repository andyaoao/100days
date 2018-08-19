# 9日目KNN分類　Day 9 KNN Classification

本日の目標は
1. データを取り込みから整理まで
2. multiple linear regressionのモデルを作成
3. モデルで予測結果を作成
4. cunfusion matrixの作成
5. SVMの等高線を描く

## Step 1: データを取り込みから整理まで
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

```
## Step 2: Logistics linear regressionのモデルを作成
```python
from sklearn.svm import SVC

# Fitting SVM to the Training set
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

```
## Step 3: モデルで予測結果を作成
```python
# 予測結果
y_pred = classifier.predict(X_test)

```
## Step 4: cunfusion matrixの作成
```python
from sklearn.metrics import confusion_matrix
# cunfusion matrixを作る
cm = confusion_matrix(y_test, y_pred)
```

## Step 5: SVMの等高線を描く
```python
# Training set を図で表現する
# day 11で作った等高線のfunction
plot_decision_regions(X_train, y_train, classifier=classifier)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Test set を図で表現する
plot_decision_regions(X_test, y_test, classifier=classifier)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```

## 補足

### 参考資料
confusion function http://www.baru-san.net/archives/141  
SVM説明 http://neuro-educator.com/ml5/  
SVM説明　simp_chinese version https://www.zhihu.com/question/21094489  
