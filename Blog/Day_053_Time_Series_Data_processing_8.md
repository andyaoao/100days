# 50日目 Time Series Data processing 7 Day48 Time Series Data processing 7

本日の目標は
1. 時系列予測の手法
2. 時系列予測手法の実装

## Step 1: Prophetの理解
メインは3手法がある。  
1. Top-Down：最上位の数字を予測したあと、比率配分で、下層の予測を決める。  
2. Bottom-Up；最小粒度の数字を予測した後、積み上げて、上位の予測を決める。
3. Middle-Out：中間層の数字を予測した後、下層は比率配分、上位は積み上げの中間手法。　　
決定時は、通常処理スピードで決める。  
下記の場合、bottom-up手法は時間がかかりすぎるため、middle-out手法にした。

## Step 2: 時系列予測手法の実装

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
