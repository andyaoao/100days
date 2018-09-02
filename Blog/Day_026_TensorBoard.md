# 26日目TensorBoardの使い方  Day26 TensorBoard

本日の目標は
1. TensorBoardのロジック理解
1. TensorBoardのコード追加
2. TensorBoardの起動

## Step 1: TensorBoardのロジック理解
TrainingのProcessを可視化できるlibrary
tensorflowのtraining processをlog fileに書き込んで、tensorboardに読み込まれる

## Step 2: TensorBoardのコード追加
```python
必要なlibraryをimport
import tensorflow as tf
import keras.backend.tensorflow_backend
import keras.callbacks
from keras.callbacks import TensorBoard

# TensorBoardに必要なfileを定義
NAME = "tensorboard_test-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# model trainingの際に、callbackに上記で定義されたtensorflow log fileを引数として渡す
train_history=model.fit(x=x_Train4D_normalize,
                        y=y_TrainOneHot,validation_split=0.2,
                        epochs=2, batch_size=300,verbose=2, callbacks=[tensorboard])
```

## Step 3: TensorBoardの起動
表示できない。色々試し中

エラー1：提示されたurlでtensorboardが起動されない  
solution：http://127.0.0.1:6006/　これで行けた

エラー2：No dashboards are active for the current data set.  
solution；tensorboardの定義は、model構築の後ということ  

起動のコマンド  
tensorboard --logdir='logs/'  


## 補足

### 参考資料
tensorboard実装  
https://www.youtube.com/watch?v=BqgTU7_cBnk&list=PLQVvvaa0QuDfhTox0AjmQ6tvTgMBZBEXN&index=4  
https://qiita.com/umesaku/items/3d1db13414498da31b57  
https://qiita.com/n_kats_/items/3de448e991069cde9940  

tensorboard起動の問題  
https://github.com/tensorflow/tensorboard/issues/1174  
https://github.com/tensorflow/tensorflow/issues/7856  
https://stackoverflow.com/questions/47113472/tensorboard-error-no-dashboards-are-active-for-current-data-set  
