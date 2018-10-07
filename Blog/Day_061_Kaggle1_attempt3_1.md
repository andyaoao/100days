# 61日目 Prophet 5 Day61 Prophet 5

本日の目標は
1. 分析案3の仮説を立てる
2. 分析案3を実装（データ処理の練習）

## Step 1: 分析案2の仮説を立てる
分析案1、2のパフォーマンスが良くないため、分析案3を立てる。

仮説：売上の変化は点数に影響がない  
分析案：店舗別商品別の点数の月推移でモデルを構築、2015/11を予測。


## Step 2: 分析案2を実装（データ処理の練習）
```python
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

# 2015/10の商品別、店舗別、月別の集計
sales = sales.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].sum()
sales = sales.unstack().unstack()
sales = sales.reset_index()
sales = sales.fillna(0)
print ("sales")
print (sales.head())

```


### 参考資料
Pandas  
https://note.nkmk.me/python-pandas-set-index/  
https://pythondatascience.plavox.info/pandas/%E8%A1%8C%E3%83%BB%E5%88%97%E3%81%AE%E9%95%B7%E3%81%95%E3%82%92%E7%A2%BA%E8%AA%8D  
https://note.nkmk.me/python-dict-keys-values-items/  
