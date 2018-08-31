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
# Avtivationは0-1の数値なので、色の範囲を0-255 -> 0-1に変換
# input layerには784 neuronsがある

# 普通の画像であれば、3色がある。今回はもう一つの
x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')

# 將數值縮小到0~1
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255

 # 把類別做Onehot encoding
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)

from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

model = Sequential()

#filter為16, Kernel size為(5,5),Padding為(same)
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1),
                 activation='relu'))

# MaxPooling size為(2,2)
model.add(MaxPooling2D(pool_size=(2, 2)))


modelmodel..addadd((Conv2DConv2D((filtersfilters==3636,,
                          kernel_size=(5,5),
                 padding='same',
                 activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

# Drop掉部分神經元避免overfitting
model.add(Dropout(0.25))


# 平坦化
 model.add(Flatten())

 model.add(Dense(128, activation='relu'))
 model.add(Dropout(0.5))
 model.add(Dense(10,activation='softmax'))
 print(model.summary())

 model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

 rain_history=model.fit(x=x_Train4D_normalize,
                        y=y_TrainOneHot,validation_split=0.2,
                        epochs=20, batch_size=300,verbose=2)

def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')
