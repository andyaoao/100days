# 46日目 Time Series Data processing 1 Day46 Time Series Data processing 1

本日の目標は
1. Kaggle Competitionの目的とデータ内容を確認
2. 仮説を立てる
3. データを読み込み

## Step 1: Kaggle Competitionの目的とデータ内容を確認

kaggle目的：2015年11月の店舗別商品別の販売点数予測
input：
1. 日別店舗別商品別の売上、点数
2. 商品グループマスタ
3. 商品マスタ
4. 店舗マスタ

個人目的：本competitionを通して、時系列分析を触ってみる

## Step 2: 仮説を立てる

月別の推移売上から、各店舗の商品別売上を予測できる  


## Step 3: データを読み込み
```python

import numpy as np
import pandas as pd
import random as rd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns


# 時系列関連library
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

# データを取り込む
sales=pd.read_csv("./Datasets/PredictFutureSales/sales_train.csv")
item_cat=pd.read_csv("./Datasets/PredictFutureSales/item_categories.csv")
item=pd.read_csv("./Datasets/PredictFutureSales/items.csv")
sub=pd.read_csv("./Datasets/PredictFutureSales/sample_submission.csv")
shops=pd.read_csv("./Datasets/PredictFutureSales/shops.csv")
test=pd.read_csv("./Datasets/PredictFutureSales/test.csv")

# データ型の確認
print (sales.info())

# 日付データ型を整理
sales.date=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
print (sales.head())




```
