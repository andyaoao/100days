# 80日目 XGBoost Day80 XGBoost

本日の目標は
1. XGBoost の理解

## Step 1: XGBoost の理解

### 概念
regression model の変形。  
複数のdecision treeに構築される。  

### additive training  
毎回新しい関数を追加するとき、前回の関数をベースで進む。  

### split finding algorithms
exact greedy algorithm:全ての可能性を計算する
approximate algorithm:可能性高い候補をリストし、その中から選出

##　参考資料
https://medium.com/@cyeninesky3/xgboost-a-scalable-tree-boosting-system-%E8%AB%96%E6%96%87%E7%AD%86%E8%A8%98%E8%88%87%E5%AF%A6%E4%BD%9C-2b3291e0d1fe  
http://hatunina.hatenablog.com/entry/2018/05/22/215959  
