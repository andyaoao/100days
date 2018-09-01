# 24日目CNN 2 Day24 Convolution Neural Network 2

本日の目標は
1. データを整理・分割
2. CNNのconvolutional layerの理解、実装

## Step 1: データを整理・分割
```python
import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

# kerasの内蔵data setを読み込み
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# インポートしたデータを確認
fig = plt.figure(figsize=(9,9))

for i in range(36):
    ax = fig.add_subplot(6, 6, i+1, xticks=[], yticks=[])
    ax.imshow(x_train[i], cmap='gist_gray')

plt.show()

# 配列の形を整理する
# 全ての画像は28*28pixel。普通の画像に3色があるが、今回は白黒のため、軸に1個追加。
# x_train.shape は画像の個数
x_Train4D=x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_Test4D=x_test.reshape(x_test.shape[0],28,28,1).astype('float32')

# Avtivationは0-1の数値なので、色の範囲を0-255 -> 0-1に変換
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255

 # outputの値をコード化
y_TrainOneHot = np_utils.to_categorical(y_train)
y_TestOneHot = np_utils.to_categorical(y_test)

```
## Step 2: CNNのconvolutional layerの理解、実装
```python

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

# モデルを作成
model = Sequential()
# input: 28*28*1
# output: 28*28*1


# convolutional layer を追加
# padding same ： ゼロパディングすることで、出力画像は入力画像と同じサイズになる。
# 16個filter それぞれの大きさは5*5
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu'))
```

## 補足

### 参考資料
Keras CNN padding  
http://ni4muraano.hatenablog.com/entry/2017/02/02/195505  
