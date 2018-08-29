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

# Functional API

from keras.engine import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop

# Modelはinputとoutputを分けている
# inputのneurons数を定義
inputs = Input(shape=(784,))

# hidden layerに対して、neuronsとactivation計算を定義
# 最後の(inputs)は１個前のlayerはどのlayerを定義
nw = Dense(512, activation='relu')(inputs)
nw = Dropout(.2)(nw)
# 現在のnwは 784 + 512 の状態
nw = Dense(512, activation='relu')(nw)
nw = Dropout(.2)(nw)
# 現在のnwは 784 + 512 + 512の状態
predictions = Dense(10, activation='softmax')(nw)
# predictionsは 784 + 512 + 512 + 10の状態

# inputとoutputを定義し、modelに渡し、コンパイル
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

# トレーニング(epochsはiteration)
history = model.fit(x_train, y_train, batch_size=128, epochs=2, verbose=1, validation_data=(x_test, y_test))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
