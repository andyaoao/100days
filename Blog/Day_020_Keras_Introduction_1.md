# 20日目Keras introduction 1 Day20 Keras introduction 1

本日の目標は
1. Kerasのロジックを理解
2. 実装してみる

## Step 1: Kerasのロジックを理解
Kerasとは、Pythonで書かれたNeural Network専門のLibraryだ。  
Kerasのbackendはいくつかある、tensorflowはその一つだ。  
2種類のnetworkがある。SequentialモデルとFunctional APIがある。

## Step 2: 実装してみる
実装エラー、調査中。

調査経過
1. keras、softmaxのパラメータにaxisが存在しないようで　→　kerasのバージョン問題？
2. バージョン確認したら、kerasは2.2.2、tensorflowは1.0だ
3. kerasは2.08にダウン、tensorflowを1.4にあげた　→　だめ
4. kerasは2.22にアップ、tensorflowを1.8にあげた　→　だめ

バージョンに問題なさそうなので、別コードで試す

## 補足

### 参考資料

keras tutorial https://qiita.com/yampy/items/706d44417c433e68db0d  
Functional APIによる実装　https://github.com/yampy/machine-learning/blob/master/keras/keras-introduction/keras_mnist_pixeldata.ipynb  
kerasのバージョン問題調査  
https://stackoverflow.com/questions/50776598/typeerror-softmax-got-an-unexpected-keyword-argument-axis    
 https://blog.csdn.net/bailianfa/article/details/80891051  
https://yuedy.com/%E3%83%95%E3%83%AD%E3%83%B3%E3%83%88%E3%82%A8%E3%83%B3%E3%83%89/329748/  
