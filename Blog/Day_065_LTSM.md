# 65日目 LTSM 1 Day65 LTSM 1

本日の目標は
1. LTSMの理解
2. 分析案3を実装（データ処理の練習）

## Step 1: LTSMの理解
RNNのひとつ、長期間のデータによる大量の演算を解消するためのほう。    
RNNとCNNの差別：CNNが扱う画像データは二次元の矩形データでしたが、音声データは可変長の時系列データです。  

LSTMにはInput Gate、Output Gate、Memory Cell、Forget Gate　四つのコンポーネントがある。  
Input Gate: featureより値をinputするとき、inputするかしないかのコントロール  
Memory Cell: 計算した値を保存  
Output Gate: 今回の計算値は次のinputとして使うかどうかをコントロール  
Forget Gate: 今回の計算値を忘れるかどうかをコントロール  


##　参考資料
https://medium.com/data-scientists-playground/lstm-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E8%82%A1%E5%83%B9%E9%A0%90%E6%B8%AC-cd72af64413a
http://gagbot.net/machine-learning/ml
