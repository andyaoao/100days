import pandas as pd
import keras
from keras.datasets import mnist
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
# Avtivationは0-1の数値なので、色の範囲を0-255 -> 0-1に変換
# input layerには784 neuronsがある
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255

# outputのカテゴリをコード化
y_train = keras.utils.np_utils.to_categorical(y_train.astype('int32'),10)
y_test = keras.utils.np_utils.to_categorical(y_test.astype('int32'),10)


# Sequential Model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop


# Sequientialモデルを実装してみた
# 作り方は、まずモデルを定義し、各オブジェクトをaddの形で順次追加する
model = Sequential()

# hidden layerの第一層を定義、512neuronsがある
# acrivation の計算方法は ReLU(rectified linear unit)とする
# 第一層なので、input_shapeを定義する必要がある
model.add(Dense(512, activation='relu', input_shape=(784,)))

# 訓練時の更新においてランダムに入力ユニットを0とする割合であり，過学習の防止
model.add(Dropout(0.2))

# hidden layerの第二層を定義、512neuronsがある
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

# output layerを定義、10 neuronsがある
# activation の計算方法はsoftmaxとする（結果は0から1に集約する）
model.add(Dense(10, activation='softmax'))

# 上記で定義されたオブジェクトをコンパイルする
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# トレーニング(epochsはiteration)
history = model.fit(x_train, y_train, batch_size=128, epochs=2, verbose=1, validation_data=(x_test, y_test))

# 結果を図で出力する
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# ロス計算結果を図で出力する
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
