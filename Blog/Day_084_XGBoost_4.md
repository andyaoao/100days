# 83日目 XGBoost Day83 XGBoost

本日の目標は
1. XGBoost のパラメータチューニング

## Step 1: XGBoost のパラメータチューニング

### min_child_weightとmax_depth
min_child_weightとmax_depthがモデルに一番影響があるので、この二つからチューニングする。  

実装コード：
```python
param_test1 = {
 # チューニングしたい範囲を決める
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
# GridSearchCVでパフォーマンス一番いいの組み合わせを探す
gsearch1 = GridSearchCV(estimator = XGBClassifier(         learning_rate =0.1, n_estimators=140, max_depth=5,
min_child_weight=1, gamma=0, subsample=0.8,             colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4,     scale_pos_weight=1, seed=27),
 param_grid = param_test1,     scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(train[predictors],train[target])
print (gsearch1.grid_scores_, gsearch1.best_params_,     gsearch1.best_score_)
```

### gamma
分岐基準を決める

実装コード：
```python
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
# GridSearchCVでパフォーマンス一番いい数字を探す
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4, min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch3.fit(train[predictors],train[target])
print (gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
```
### subsampleとcolsample_bytree
subsampleは全体サンプリングの比率。  
colsample_bytreeは各列のサンプリングの比率。  


実装コード：
```python
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}
# GridSearchCVでパフォーマンス一番いい数字を探す
gsearch3 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=4, min_child_weight=6, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1,seed=27), param_grid = param_test3, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

gsearch3.fit(train[predictors],train[target])
print (gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_)
```


## 参考資料
Sequential Decision Tree Building：　https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/456267/  
http://zhanpengfang.github.io/418home.html  
parameter: https://blog.csdn.net/han_xiaoyang/article/details/52665396  
