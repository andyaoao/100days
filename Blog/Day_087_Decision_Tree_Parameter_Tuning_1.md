# 87日目 Decision Tree Parameter Tuning 1

本日の目標は
1. 決定木のパラメータチューニングプロセス

## Step 1: 決定木のパラメータチューニングプロセス

### 0. 事前準備
決定木手法を適用する前、欠損値の処理、カテゴリfeatureの処理  
＊feature scaling通常必要なし

### 1. Default
Defaultのパラメータ値でfit  

### 2. AUC(Area under curve)
binary classification problemのであれば、AUC(Area under curve)は評価のツールとして使える  

### 3. max_depth
決定木の最大深さを決める。  
チューニング方法：特定の範囲内でテストし、AUCをベースでパフォーマンスを把握する(overfittingの状況も)　　
大きければ大きいほど、精度が上がるが、overfittingの可能性も上がる。　　  

```python
# 1から32までをテスト対象とする
max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
   dt = DecisionTreeClassifier(max_depth=max_depth)
   dt.fit(x_train, y_train)

   # training set の　auc
   train_pred = dt.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)

   # testing set の　auc
   y_pred = dt.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)

# training set と testing set のAUCを描画する
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, ‘b’, label=”Train AUC”)
line2, = plt.plot(max_depths, test_results, ‘r’, label=”Test AUC”)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(‘AUC score’)
plt.xlabel(‘Tree depth’)
plt.show()
```

### 4. min_samples_split
node splitの最小サンプル数を決める（数量 or パーセンテージ）。  
チューニング方法：特定の範囲内でテストし、AUCをベースでパフォーマンスを把握する。
大きければ大きいほど、splitの困難度が上がり、精度が落ちる。  

```python
# 10%から100%までをテスト対象とする
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
   dt = DecisionTreeClassifier(min_samples_split=min_samples_split)
   dt.fit(x_train, y_train)

   # training set の　auc
   train_pred = dt.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds =    roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)

   # testing set の　auc
   y_pred = dt.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)

# training set と testing set のAUCを描画する
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_splits, train_results, ‘b’, label=”Train AUC”)
line2, = plt.plot(min_samples_splits, test_results, ‘r’, label=”Test AUC”)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(‘AUC score’)
plt.xlabel(‘min samples split’)
plt.show()

```

### 5. min_samples_leaf
一つのnodeの最小サンプル数を決める（数量 or パーセンテージ）。  
チューニング方法：特定の範囲内でテストし、AUCをベースでパフォーマンスを把握する。
大きければ大きいほど、splitの困難度が上がり、精度が落ちる。  

```python
# 10%から50%までをテスト対象とする
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
   dt = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
   dt.fit(x_train, y_train)
   train_pred = dt.predict(x_train)

   # training set の　auc
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)

   # testing set の　auc
   y_pred = dt.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)

# training set と testing set のAUCを描画する
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(min_samples_leafs, train_results, ‘b’, label=”Train AUC”)
line2, = plt.plot(min_samples_leafs, test_results, ‘r’, label=”Test AUC”)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(‘AUC score’)
plt.xlabel(‘min samples leaf’)
plt.show()
```

### 6. max_features
最適モデルの最大feature数を決める。  
チューニング方法：特定の範囲内でテストし、AUCをベースでパフォーマンスを把握する。
大きければ大きいほど、精度が上がるが、overfittingの可能性も上がる。  


```python
# 1個から全featureの個数をテスト対象とする
max_features = list(range(1,train.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
   dt = DecisionTreeClassifier(max_features=max_feature)
   dt.fit(x_train, y_train)

   # training set の　auc   
   train_pred = dt.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)

   # testing set の　auc
   y_pred = dt.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)

# training set と testing set のAUCを描画する
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_features, train_results, ‘b’, label=”Train AUC”)
line2, = plt.plot(max_features, test_results, ‘r’, label=”Test AUC”)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(‘AUC score’)
plt.xlabel(‘max features’)
plt.show()
```

## 参考資料
https://medium.com/@mohtedibf/indepth-parameter-tuning-for-decision-tree-6753118a03c3  
https://www.facebook.com/notes/python-scikit-learn/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92_ml_decisiontreeclassifier%E6%B1%BA%E7%AD%96%E6%A8%B9/802425689936465/  
