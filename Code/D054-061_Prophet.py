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
# print (sales.head())

# 日別の売り上げデータを月別に集計する(日付は範囲、売上は平均、個数は合算)
monthly_sales=sales.groupby(["date_block_num","shop_id","item_id"])[
    "date","item_price","item_cnt_day"].agg({"date":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})

# print (monthly_sales.head(20))

# date block num(年月)をベースで全社の販売点数を積上げる
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
# ts.astype('float')
print (ts.head())

# prophetが受け入れるデータ形は、日付(ds)と値(y)
ts.index=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')
ts=ts.reset_index()
ts.columns=['ds','y']
print (ts.head())


#時系列モデルを定義
# パラメータは、年周期があること
model = Prophet('linear', yearly_seasonality=True)

model.fit(ts)

# 2017/11を予測
future = model.make_future_dataframe(periods = 1, freq = 'MS')
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# 2015/10の売上配分を計算。
# 予測を配分。
