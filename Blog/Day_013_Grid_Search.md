# 13日目Grid Searchでパフォーマンスチューニング　Day 13 Grid Search

本日の目標は
1. SVMのKernel Trick
2. Grid Searchの使い方

## Step 1: SVMのKernel Trick

Linear kernel  : K(x,y) = xTy → linear separable dataset  
Polynomial :  K(x,y) = (xty+c)d → non-linear separable dataset
Sigmoid : K(x,y) = tanh(γxTy+c) → Logistic Regression
Radial basis function(RBF)： exp(-γ||x-y||2) → non-linear separable dataset

## Step 2: Grid Searchの使い方
```python
from sklearn.svm import SVC

# 探索するパラメータを設定
param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 1 0, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]
# 評価関数を指定
scores = ['accuracy', 'precision', 'recall']

# 各評価関数ごとにグリッドサーチを行う
score in scores:
    print score
    clf = GridSearchCV(SVC(C=1), param_grid, cv=5, scoring=score, n_jobs=-1)  # n_jobs: 並列計算を行う（-1 とすれば使用PCで可能な最適数の並列処理を行う）
    clf.fit(X_train, y_train)

    print clf.best_estimator_  # 最適なパラメータを表示

    for params, mean_score, all_scores in clf.grid_scores_:
        print "{:.3f} (+/- {:.3f}) for {}".format(mean_score, all_scores.std() / 2, params)

    # 最適なパラメータのモデルでクラスタリングを行う
    y_true, y_pred = y_test, clf.predict(X_test)
    print classification_report(y_true, y_pred)  # クラスタリング結果を表示
    print confusion_matrix(y_true, y_pred)       # クラスタリング結果を表示
```


## 補足

### 参考資料
Kernel Trickの説明 https://medium.com/@chih.sheng.huang821/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-kernel-%E5%87%BD%E6%95%B8-47c94095171  
Kernel Trickの説明 https://chtseng.wordpress.com/2017/02/04/support-vector-machines-%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%A9%9F/  
Grid Searchの実現 https://qiita.com/arata-honda/items/8d08f31aa7d7cbae4c91  
Grid Searchの実現 https://blog.amedama.jp/entry/2017/09/05/221037  
