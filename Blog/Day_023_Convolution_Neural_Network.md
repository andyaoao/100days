# 23日目CNN 1 Day23 Convolution Neural Network 1

本日の目標は
1. NNとの違い
2. CNNのロジック理解

## Step 1: NNとの違い
画像識別を例として、普通のNNだと、少し位置がずれていたりすると判定に大きな影響が出る。  
CNNはある程度の広さの領域をまとめて入力にすることができれば、より精度の高い判定ができそう。  
つまり、画像上にフィルタと呼ばれる小領域をとり、これを1つの特徴量として圧縮する。  

普通のNN：  
input layer -> hidden layer -> output layer
CNN：  
input layer -> Convolution layer -> pool layer -> output layer(fully connected layer)

## Step 2: CNNのロジック理解

Convolutional Layer: 特徴量の畳み込みを行う層(フィルタを作る)  
　フィルタの数(K): 使用するフィルタの数。大体は2の累乗の値がとられる(32, 64, 128 ...)  
　フィルタの大きさ(F): 使用するフィルタの大きさ  
　フィルタの移動幅(S): フィルタを移動させる幅  
　パディング(P): 画像の端の領域をどれくらい埋めるか  
Pooling Layer: レイヤの縮小を行い、扱いやすくするための層  
　Poolingの方式は、max pooling。各領域内の最大値をとって圧縮を行う方法  
Fully Connected Layer: 特徴量から、最終的な判定を行う層  

CNNの基本的な構成としては、以下のパターンが多いそう  
(Convolution * N + (Pooling)) * M + Fully Connected * K

## 補足

### 参考資料
CNNロジック  
https://qiita.com/icoxfog417/items/5fd55fad152231d706c2  
http://cs231n.github.io/convolutional-networks/  
