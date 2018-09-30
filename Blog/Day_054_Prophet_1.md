# 50日目 Prophet 1 Day48 Prophet 1

本日の目標は
1. Kaggle Competitionの分析方法検討（分析案1）
2. 分析案1を実装

## Step 1: Kaggle Competitionの仮説と分析案を検討
下記のKaggle Competitionを選定：  
https://www.kaggle.com/c/competitive-data-science-predict-future-sales  

kaggle目的：2015年11月の店舗別商品別の販売点数予測
input：
1. 日別店舗別商品別の売上、点数
2. 商品グループマスタ
3. 商品マスタ
4. 店舗マスタ

仮説：売上の変化は点数に影響がない  
分析案：全社の点数の月推移でモデルを構築、2015/11を予測。その後、2015/10の売上配分をベースで、予測を配分する

## Step 2: 分析案1を実装

```python
# bottom up method
# 最小粒度のデータを予測した後、積み上げて、店舗、全体の売り上げ予測となる
# 店舗、商品別の売上データを整理する
total_sales=sales.groupby(['date_block_num'])["item_cnt_day"].sum()
dates=pd.date_range(start = '2013-01-01',end='2015-10-01', freq = 'MS')

total_sales.index=dates
total_sales.head()

import time
start_time=time.time()

# 上記で整理したデータをモデル構築する
forecastsDict = {}
for node in range(len(monthly_sales)):
    nodeToForecast = pd.concat([monthly_sales.iloc[:,0], monthly_sales.iloc[:, node+1]], axis = 1)

    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})
    growth = 'linear'
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    future = m.make_future_dataframe(periods = 1, freq = 'MS')
    forecastsDict[node] = m.predict(future)
    if (node== 10):
        end_time=time.time()
        print("forecasting for ",node,"th node and took",end_time-start_time,"s")
        break

# 10 nodeを予測するには30秒以上かかったので、この方法をやめるべき

# Middle out method
# 店舗別の売上データを整理する
monthly_shop_sales=sales.groupby(["date_block_num","shop_id"])["item_cnt_day"].sum()
# get the shops to the columns
monthly_shop_sales=monthly_shop_sales.unstack(level=1)
monthly_shop_sales=monthly_shop_sales.fillna(0)
monthly_shop_sales.index=dates
monthly_shop_sales=monthly_shop_sales.reset_index()
monthly_shop_sales.head()

# 上記で整理したデータをモデル構築する
start_time=time.time()

forecastsDict = {}
for node in range(len(monthly_shop_sales)):

    nodeToForecast = pd.concat([monthly_shop_sales.iloc[:,0], monthly_shop_sales.iloc[:, node+1]], axis = 1)
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[0] : 'ds'})
    nodeToForecast = nodeToForecast.rename(columns = {nodeToForecast.columns[1] : 'y'})
    growth = 'linear'
    m = Prophet(growth, yearly_seasonality=True)
    m.fit(nodeToForecast)
    future = m.make_future_dataframe(periods = 1, freq = 'MS')
    forecastsDict[node] = m.predict(future)

#predictions = np.zeros([len(forecastsDict[0].yhat),1])
nCols = len(list(forecastsDict.keys()))+1
for key in range(0, nCols-1):
    f1 = np.array(forecastsDict[key].yhat)
    f2 = f1[:, np.newaxis]
    if key==0:
        predictions=f2.copy()
       # print(predictions.shape)
    else:
       predictions = np.concatenate((predictions, f2), axis = 1)
```


### 参考資料
Prophet  
https://www.slideshare.net/hoxo_m/prophet-facebook-76285278  
https://questpm.hatenablog.com/entry/fbprophet  
