# 83日目 XGBoost Day83 XGBoost

本日の目標は
1. XGBoost のSequential Decision Tree Building
2. XGBoost のパラメータチューニング

## Step 1: XGBoost のSequential Decision Tree Building

ツリーの分け目を決めるため、featureの数字をソートする必要がある。  
featureをソートした後、各leaf nodeでも使える。
パラ実行というのは、featureをパラで実行すること。   

## Step 2: XGBoost のパラメータチューニング

### general parameters
booster:  
 -> gbtree：ツリーモデル
 -> gbliner：線型モデル

### booster parameters
eta:  
  -> learning rateのようなパラメータ 普通は0.3  
min_child_weight:  
  -> 一番小さいサンプルのweightの合計（大きければ大きいほど過学習を予防できる）
max_depth:
　-> ツリーの深さ
gamma:
　-> 分岐を作る基準は、分岐したら、cost functionの減った分がgammaの値より大きい場合のみ分岐を作る。  

## 参考資料
Sequential Decision Tree Building：　https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/456267/  
http://zhanpengfang.github.io/418home.html  
parameter: https://blog.csdn.net/han_xiaoyang/article/details/52665396  
