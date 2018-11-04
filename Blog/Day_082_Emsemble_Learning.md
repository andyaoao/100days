# 82日目 Ensemble learning Day82 Ensemble learning

本日の目標は
1. Ensemble learning (アンサンブル学習)
2. Stacked modelの構造

## Step 1: Ensemble learning (アンサンブル学習)

### bias and variance
バイアス　bias：実際値と予測値との誤差の平均のことで、値が小さいほど予測値と真の値の誤差が小さいということになります。  
バリアンス　variance：予測値がどれだけ散らばっているかを示す度合いのことで、値が小さいほど予測値の散らばりが小さいということになります。  

高バイアス：訓練不足  
高バリアンス：過学習  
バイアスとバリアンスはトレードオフの関係にある  

### 3つの分類手法

バギング Bootstrap Aggregating：学習データのサンプル抽出をわざとダブらせる方法。1-6のサンプルの中、それぞれ3倍用意し、18サンプルをランダム3つのモデルで学習させる。  
ブースティング Boosting：連続的に同じモデルを訓練させる。  
 AdaBoost (Adaptive Boosting)：１回目で訓練できたモデルの中、間違ったobservationに対して、weightを追加し、次の訓練に入る。  
 Gradient Boosting：１回目で訓練できたモデル後、residualに対して、2回目の訓練に入る。  
スタッキング　Stacking：2あるいはmore level prediction。概念として、base model複数あり、それぞれの予測結果をstack model(level 2)のfeatureとして投入する。  

## Step 2: Stacked modelの構造

1. training set を　5つのfoldを作る
2. その中の4foldをtraining setとして、5つ目のfoldを予測する（5回の予測結果がある。複数のlevel-1モデルがある場合、複数の予測結果がある)
3. 予測結果をfeatureとして、level-2のモデルを作成
4. stacked model(level-2 model)で結果を予測


# 参考資料
boosting : https://medium.com/greyatom/a-quick-guide-to-boosting-in-ml-acf7c1585cb5  
stacking : https://blogs.sas.com/content/subconsciousmusings/2017/05/18/stacked-ensemble-models-win-data-science-competitions/#prettyPhoto   
build stacking model : http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/  
