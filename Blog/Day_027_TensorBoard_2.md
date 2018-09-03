# 27日目TensorBoardの使い方 2 Day26 TensorBoard 2

本日の目標は
1. TensorBoardでベストプラクティスを探す


## Step 1: TensorBoardでベストプラクティスを探す
同じlogフォルダの下に、全てのlogを一つのboardで確認することができる。  
上記の利点を利用し、ベストプラクティスを探す。  

```python
# 試した組み合わせを定義
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

# 各組み合わせをベースでlog fileを作成する
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

            # log fileをcallbacksで書き出し
            train_history=model.fit(x=x_Train4D_normalize,
                                    y=y_TrainOneHot,validation_split=0.2,
                                    epochs=2, batch_size=300,verbose=2, callbacks=[tensorboard])

```


## 補足

### 参考資料
tensorboard実装  
https://pythonprogramming.net/tensorboard-optimizing-models-deep-learning-python-tensorflow-keras/  
