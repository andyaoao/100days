import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

# day 23 start

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

# day 24 start

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
# input: 28*28*1
# output: 28*28*16

# pooling layerを追加
# MaxPooling sizeは2として、移動のpixel数は2とする
model.add(MaxPooling2D(pool_size=(2, 2)))
# input: 28*28*16
# output: 14*14*16

# convolutional layer を追加
# 36個filter それぞれの大きさは5*5
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))

# input: 14*14*16
# output: 14*14*36

# pooling layerを追加
# MaxPooling sizeは2として、移動のpixel数は2とする
model.add(MaxPooling2D(pool_size=(2, 2)))
# input: 14*14*36
# output: 7*7*36

# 過学習を防ぐ
model.add(Dropout(0.25))

# 平坦化
model.add(Flatten())
# input: 14*14*36
# output: 7*7*36

model.add(Dense(128, activation='relu'))
# input: 7*7*36
# output: 1764

model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
# input: 1764
# output: 10

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

train_history=model.fit(x=x_Train4D_normalize,
                        y=y_TrainOneHot,validation_split=0.2,
                        epochs=3, batch_size=300,verbose=2)

def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')
