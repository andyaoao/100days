import numpy as np
import pandas as pd
import random as rd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import time


# 時系列関連library
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
from fbprophet import Prophet

# データを取り込む
sales=pd.read_csv("./Datasets/PredictFutureSales/sales_train.csv")
item_cat=pd.read_csv("./Datasets/PredictFutureSales/item_categories.csv")
item=pd.read_csv("./Datasets/PredictFutureSales/items.csv")
sub=pd.read_csv("./Datasets/PredictFutureSales/sample_submission.csv")
shops=pd.read_csv("./Datasets/PredictFutureSales/shops.csv")
test=pd.read_csv("./Datasets/PredictFutureSales/test.csv")

# データ型の確認
# print (sales.info())

# 日付データ型を整理
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))

# 前月のデータを抽出
# sales_201510 = sales[sales["date_block_num"] == 33]
# # as_index=Falseは実テーブルのように、一レコード一行で格納
# # 2015/10の商品別、店舗別の集計
# sales_201510 = sales_201510.groupby(["date_block_num","item_id","shop_id"], as_index=False).sum()
# print ("sales_201510_item")
# print (sales_201510.head())
#
# # 比率を計算する
# # sum関数で店舗内のpercentageを算出
# sales_201510["percentage"] = sales_201510["item_cnt_day"] / sales_201510["item_cnt_day"].sum()
# print ("sales_201510 percentage")
# print (sales_201510.head())
#
# # sum関数で店舗内のpercentageを算出
# sales_201510 = sales_201510.groupby(["date_block_num","item_id"], as_index=False).sum()
# print ("sales_201510 percentage")
# print (sales_201510.head())
#
#
# # date block num(年月)をベースで全社の販売点数を積上げる
# ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
# print ("ts")
# print (ts.head())
#


# 店舗別の売上データを整理する
monthly_shop_sales=sales.groupby(["date_block_num","shop_id"])["item_cnt_day"].sum()
# unstack関数で、行を列に変換(level 1 )
monthly_shop_sales=monthly_shop_sales.unstack(level=1)
monthly_shop_sales=monthly_shop_sales.fillna(0)
# datesをindexにする方法を調査中
# monthly_shop_sales.index=dates
# monthly_shop_sales=monthly_shop_sales.reset_index()
print ("monthly_shop_sales")
print (monthly_shop_sales.head())


# 上記で整理したデータをモデル構築する
start_time=time.time()

forecastsDict = {}
for node in range(len(monthly_shop_sales)):

    nodeToForecast = pd.concat([monthly_shop_sales.iloc[:,0], monthly_shop_sales.iloc[:, node+1]], axis = 1)
    print ("nodeToForecast")
    print (nodeToForecast.head())
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})
    growth = 'linear'
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    future = m.make_future_dataframe(periods = 1, freq = 'MS')
    forecastsDict[node] = m.predict(future)

#predictions = np.zeros([len(forecastsDict[0].yhat),1])
# nCols = len(list(forecastsDict.keys()))+1
# for key in range(0, nCols-1):
#     f1 = np.array(forecastsDict[key].yhat)
#     f2 = f1[:, np.newaxis]
#     if key==0:
#         predictions=f2.copy()
#        # print(predictions.shape)
#     else:
#        predictions = np.concatenate((predictions, f2), axis = 1)

#
#
#
# # prophetが受け入れるデータ形は、日付(ds)と値(y)
# ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
# ts=ts.reset_index()
# # 列名を修正する
# ts.columns=['ds','y']
# print ("before modeling")
# print (ts.head())
#
# #時系列モデルを定義
# # パラメータは、年周期があること
# model = Prophet('linear', yearly_seasonality=True)
# model.fit(ts)
#
# # 2017/11を予測
# future = model.make_future_dataframe(periods = 5, freq = 'MS')
# forecast = model.predict(future)
# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#
# print ("forecast")
# print (forecast.head())
#
# # 5期を予測したが、2018/11はその第1期
# sales_201510["result"] = sales_201510["percentage"] * forecast["yhat"][0]
# print ("calculation")
# print (sales_201510.head())
#
# submission = pd.merge(test, sales_201510, left_on=["shop_id", "item_id"], right_on=["shop_id", "item_id"], how='left')
# print ("submission before delete")
# print (submission.head())
# submission_new = submission.drop(['date_block_num', 'shop_id', 'item_id','item_price','item_cnt_day', 'percentage'], axis=1)
# submission_new = submission_new.fillna(0)
# print ("submission after delete")
# print (submission_new.head())
#
# submission_new.columns=['ID','item_cnt_month']
# print ("submission")
# print (submission_new.head())
#
# # csvに書き出し
# submission_new.to_csv("./Datasets/PredictFutureSales/submission.csv", index=False)
#
# # 2015/10の売上配分を計算。
# # 予測を配分。
