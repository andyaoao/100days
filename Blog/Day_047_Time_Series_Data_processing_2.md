# 46日目 Time Series Data processing 2 Day46 Time Series Data processing 2

本日の目標は
1. 季節調整の理解
2. 季節調整の実装
3. 定常過程の理解
4. 定常過程のテスト

## Step 1: 季節調整
季節調整は時系列データの中、季節成分を除く手法。  
時系列のデータaddictive modelで解説：観測値 = トレンド成分 + 季節成分 + 循環成分 + ノイズ成分  
時系列のデータmultiplicative modelで解説：観測値 = トレンド成分 * 季節成分 * 循環成分 * ノイズ成分  

sm.tsa.seasonal_decomposeの関数で、上記を実現できる。  

## Step 2: 季節調整の実装

```python
import statsmodels.api as sm
# Additive model
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="additive")
fig = res.plot()

# multiplicative
res = sm.tsa.seasonal_decompose(ts.values,freq=12,model="multiplicative")
fig = res.plot()
```

## Step 3: 定常過程の理解

時間や位置によって確率分布が変化しない確率過程。  
時系列のデータに、定常過程をもつデータの定義：
1. データの移動平均は、時間と関係性がない（時間の関数ではない）
2. データの移動分散は、時間と関係性がない（時間の関数ではない）
1. 任意2データポイントの共分散は、時間と関係性がない（時間の関数ではない）

## Step 4: 定常過程のテスト

```python
# Stationarity tests
def test_stationarity(timeseries):

    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(ts)
```
