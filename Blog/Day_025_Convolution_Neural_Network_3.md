# 25日目CNN 3 Day25 Convolution Neural Network 3

本日の目標は
1. CNNのpooling layerの理解、実装(各layerのinput outputを理解)
2. CNNでtraining、予測を出力

## Step 1: CNNのpooling layerの理解、実装(各layerのinput outputを理解)
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

# モデルをコンパイル
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
```

## Step 2: CNNでtraining、予測を出力
```python

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
```

## 補足

### 参考資料
Keras CNN 実装  
https://www.hksilicon.com/articles/1410193  
http://tekenuko.hatenablog.com/entry/2017/07/23/195321  
https://qiita.com/sasayabaku/items/9e376ba8e38efe3bcf79  
https://qiita.com/icoxfog417/items/5fd55fad152231d706c2  
