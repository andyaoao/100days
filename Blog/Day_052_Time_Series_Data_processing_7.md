# 50日目 Time Series Data processing 7 Day48 Time Series Data processing 7

本日の目標は
1. Prophetの理解
2. Prophetを実装

## Step 1: Prophetの理解
additive時系列モデルをベースで開発されたlibrary。  
目的は、パラメータの設定を簡単にする。  
調整できるパラメータ：  
1. トレンドの線形、非線形
2. 変化点
3. 週周期、月周期
4. イベント

## Step 2: ARMAモデルで予測

```python
from fbprophet import Prophet

# prophetが受け入れるデータ形は、日付(ds)と値(y)
ts.columns=['ds','y']

#時系列モデルを定義
# パラメータは、年周期があること
model = Prophet( yearly_seasonality=True)

model.fit(ts)

# 5期間、頻度は月の初日(month start)
future = model.make_future_dataframe(periods = 5, freq = 'MS')  
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```


### 参考資料
Prophet  
https://www.slideshare.net/hoxo_m/prophet-facebook-76285278  
https://questpm.hatenablog.com/entry/fbprophet  
