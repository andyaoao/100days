# 48日目 Time Series Data processing 3 Day48 Time Series Data processing 3

本日の目標は
1. 定常過程まで調整
2. 定常過程の時系列モデル

## Step 1: 定常過程まで調整
```python
from pandas import Series as Series

# トレンド成分を削除するfunction
# 各期との差分を計算
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# 調整前
ts=sales.groupby(["date_block_num"])["item_cnt_day"].sum()
ts.astype('float')
plt.figure(figsize=(16,16))
plt.subplot(311)
plt.title('Original')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.plot(ts)

# 1 interval差分調整後
plt.subplot(312)
plt.title('After De-trend')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts)
plt.plot(new_ts)
plt.plot()

# 12 interval差分調整後
plt.subplot(313)
plt.title('After De-seasonalization')
plt.xlabel('Time')
plt.ylabel('Sales')
new_ts=difference(ts,12)
plt.plot(new_ts)
plt.plot()

# 調整後もう一回テストした結果
test_stationarity(new_ts)
# テストの結果、p-value < 0.01 , 定常過程
```
## Step 2: 定常過程の時系列モデル
定常過程の時系列モデル
1. AR(Autogressive)自己回帰モデル：「p個以前の過去の値」で生成したモデル  
2. MA(Moving Average Means)移動平均モデル：q個前までの過去のノイズの重み付き和と現在のノイズ、これに平均値を足しあわせたもの(今期と、過去N)  
3. ARMAモデル：「p個以前の過去の値」と「q個以前のノイズの値」によって現在の値を記述することができるモデル  


### 参考資料
AR/MAモデル  
http://muddydixon.hatenablog.com/entry/2013/03/23/050131
