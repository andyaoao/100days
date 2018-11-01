# 81日目 XGBoost Day81 XGBoost

本日の目標は
1. XGBoost の実装

## Step 1: XGBoost の実装
```python
# 必要なlibraryをimport
from xgboost import XGBClassifier

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb.score(X_test, y_test)

from sklearn.model_selection import RandomizedSearchCV
# パラメータのリストを設定
gbm_param_grid = {
    'n_estimators': range(5,20), # ツリーの個数
    'max_depth': range(6,20), # ツリーの深さ
    'learning_rate': [.4, .45, .5, .55, .6],
    'colsample_bytree': [.6, .7, .8, .9, 1],
    'min_child_weight':range(1,6,2)
    }

xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid,
                   estimator=gbm,
                   scoring="accuracy",verbose=1,
                   n_iter=50,
                   cv=4) # cross validation

# Fit randomized_mse to the data
xgb_random.fit(X, y)

# # Print the best parameters and lowest RMSE
# print("Search log: ", xgb_random.grid_scores_)
# print("Best parameters found: ", xgb_random.best_params_)
# print("Best accuracy found: ", xgb_random.best_score_)

# classifierのimport
from xgboost import XGBClassifier
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=10, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
xgb.fit(X_train, y_train)
xgb.score(X_test, y_test

```

##　参考資料
パラメター設定方法：　https://blog.csdn.net/han_xiaoyang/article/details/52665396  
