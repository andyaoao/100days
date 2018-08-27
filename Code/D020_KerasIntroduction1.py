import numpy as np
# kerasより必要なlibraryをimport
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt

# kerasの内蔵data setを読み込み
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# print (X_train.shape)
# plt.imshow(X_train[0])

# 配列の形を整理する
# Avtivationは0-1の数値なので、色の範囲を0-255 -> 0-1に変換
X_train = X_train.reshape(60000, 784) / 255
X_test = X_test.reshape(10000, 784) / 255

# outputのカテゴリをコード化
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# modelの定義
model = Sequential([
        Dense(512, input_shape=(784,)),
        Activation('sigmoid'),
        Dense(10),
        Activation('softmax')
    ])

# 設定したモデルをコンパイル
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

# 学習処理の実行
model.fit(X_train, y_train, batch_size=200, verbose=1, epochs=20, validation_split=0.1)

# 予測
score = model.evaluate(X_test, y_test, verbose=1)
print('test accuracy : ', score[1])

# 実行エラー、調査中
