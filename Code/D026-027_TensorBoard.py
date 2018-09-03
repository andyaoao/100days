import numpy as np
import pandas as pd
import keras
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

import tensorflow as tf
import keras.backend.tensorflow_backend
import keras.callbacks
from keras.callbacks import TensorBoard
import time

# day 26 start

# NAME = "tensorboard_test-{}".format(int(time.time()))

# kerasの内蔵data setを読み込み
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# インポートしたデータを確認
# fig = plt.figure(figsize=(9,9))
#
# for i in range(36):
#     ax = fig.add_subplot(6, 6, i+1, xticks=[], yticks=[])
#     ax.imshow(x_train[i], cmap='gist_gray')
#
# plt.show()


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

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

# day 27 start

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))

            # モデルを作成
            model = Sequential()

            # conv_layerの第１層
            model.add(Conv2D(filters=16,
                             kernel_size=(5,5),
                             padding='same',
                             input_shape=(28,28,1),
                             activation='relu'))

            model.add(MaxPooling2D(pool_size=(2, 2)))

            # conv_layerのlayer数をloop(-1の理由は第１層は上にある)
            for l in range(conv_layer-1):
                model.add(Conv2D(filters=16,
                                 kernel_size=(5,5),
                                 padding='same',
                                 input_shape=(28,28,1),
                                 activation='relu'))

                model.add(MaxPooling2D(pool_size=(2, 2)))

            # 平坦化
            model.add(Flatten())

            # dense_layerとdense sizesをloop)
            for l in range(dense_layer):
                model.add(Dense(layer_sizes, activation='relu'))

            # 最後にで抽出
            model.add(Dense(10,activation='softmax'))

            # TensorBoardを定義する
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
            train_history=model.fit(x=x_Train4D_normalize,
                                    y=y_TrainOneHot,validation_split=0.2,
                                    epochs=2, batch_size=300,verbose=2, callbacks=[tensorboard])

# def show_train_history(train_acc,test_acc):
#     plt.plot(train_history.history[train_acc])
#     plt.plot(train_history.history[test_acc])
#     plt.title('Train History')
#     plt.ylabel('Accuracy')
#     plt.xlabel('Epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#
# show_train_history('acc','val_acc')
