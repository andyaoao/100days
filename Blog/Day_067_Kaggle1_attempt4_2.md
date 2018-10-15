# 65日目 LTSM 1 Day65 LTSM 1

本日の目標は
1. 分析案4を考案
2. 分析案4を実装して見る

## Step 1: 時間ー文字列の処理
strptime：文字列を時間にparse  
```python
import time
import datetime

import numpy as np
import pandas as pd

# 機械学習系のlibrary
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import gc

# Viz
import matplotlib.pyplot as plt

# データを取り込む
sales=pd.read_csv("./Datasets/PredictFutureSales/sales_train.csv", parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
item_cat=pd.read_csv("./Datasets/PredictFutureSales/item_categories.csv")
item=pd.read_csv("./Datasets/PredictFutureSales/items.csv")
sub=pd.read_csv("./Datasets/PredictFutureSales/sample_submission.csv")
shops=pd.read_csv("./Datasets/PredictFutureSales/shops.csv")
test=pd.read_csv("./Datasets/PredictFutureSales/test.csv")

# 店舗商品ごとの売上、点数(合計)
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
print ("df")
print (df.head())
df = df[['date','item_id','shop_id','item_cnt_day']]
df["item_cnt_day"].clip(0.,20.)
df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()
print ("df after clip")
print (df.head())

# 点数とテストセットの形に整理する
test_sales = pd.merge(test,df,on=['item_id','shop_id'], how='left').fillna(0)
test_sales = test_sales.drop(labels=['ID','item_id','shop_id'],axis=1)

# 店舗商品ごとの売上（平均）
scaler = MinMaxScaler(feature_range=(0, 1))
sales["item_price"] = scaler.fit_transform(sales["item_price"].values.reshape(-1,1))
df2 = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).mean().reset_index()
df2 = df2[['date','item_id','shop_id','item_price']].pivot_table(index=['item_id','shop_id'], columns='date',values='item_price',fill_value=0).reset_index()
print ("df2")
print (df2.head())

# 売上とテストセットの形に整理する
price = pd.merge(test,df2,on=['item_id','shop_id'], how='left').fillna(0)
price = price.drop(labels=['ID','item_id','shop_id'],axis=1)

print ("test_sales")
print (test_sales)

# training set の　y は　最新の店舗商品点数データとする
y_train = test_sales["2015-10"]
print ("y_train")
print (y_train)

# training set の x は　最新以外のデータ
x_sales = test_sales.drop(labels=['2015-10'],axis=1)

# dataframeをarrayにrechapeする
# 行は店舗商品数214200、列は時間軸33、階数は1(後ほどpriceを追加したいから、3Dを定義する)
x_sales = x_sales.values.reshape((x_sales.shape[0], x_sales.shape[1], 1))
print ("x_sales")
print (x_sales)

x_prices = price.drop(labels=['2015-10'],axis=1)
x_prices= x_prices.values.reshape((x_prices.shape[0], x_prices.shape[1], 1))
print ("x_price")
print (x_prices)

# 点数と売上のarrayを繋げる
X = np.append(x_sales,x_prices,axis=2)
print ("X")
print (X)

# training set の y もnumpy のarrayにreshapeする
y = y_train.values.reshape((214200, 1))
print("Training Predictor Shape: ",X.shape)
print("Training Predictee Shape: ",y.shape)
del y_train, x_sales; gc.collect()

# test set をnumpy のarrayにreshapeする
# 最初の日付をtesting set から除外
test_sales = test_sales.drop(labels=['2013-01'],axis=1)
x_test_sales = test_sales.values.reshape((test.shape[0], test.shape[1], 1))
x_test_prices = price.drop(labels=['2013-01'],axis=1)
x_test_prices = x_test_prices.values.reshape((x_test_prices.shape[0], x_test_prices.shape[1], 1))

# Combine Price and Sales Df
X_test = np.append(x_test_sales,x_test_prices,axis=2)
del x_test_sales,x_test_prices, price; gc.collect()
print("Test Predictor Shape: ",X_test.shape)

print("Modeling Stage")
# Define the model layers
model_lstm = Sequential()
model_lstm.add(LSTM(16, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model_lstm.add(Dropout(0.5))
model_lstm.add(LSTM(32))
model_lstm.add(Dropout(0.5))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer="adam", loss='mse', metrics=["mse"])
print(model_lstm.summary())

# Train Model
print("\nFit Model")
VALID = True
LSTM_PARAM = {"batch_size":128,
              "verbose":2,
              "epochs":10}
```

## Step 2: 分析案5を考案
priceは考えていないから、LTSMにprice要素を追加


##　参考資料
https://docs.pyq.jp/help/quest/quest_help_strftime_strptime.html  
