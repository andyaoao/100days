# 70日目 Feature Engineering 1 Day70 Feature Engineering 1

本日の目標は
1. Two-level model 実装してみる


## Step 1: Feature Engineeringの簡単概念
```python
Validation = False
reduce_size = False
num_first_level_models = 3
SEED = 0

import time

start_time = time.time()

import pandas as pd
import numpy as np
import gc
from tqdm import tqdm

pd.set_option('display.max_rows', 99)
pd.set_option('display.max_columns', 50)

import warnings
warnings.filterwarnings('ignore')

# Data path
data_path = './Datasets/PredictFutureSales'
submission_path = './Datasets/PredictFutureSales/'

def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols =   [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int16)
    return df

# 0. Load data ----------------------------------------------------------------

# print('%0.2f min: Start loading result'%((time.time() - start_time)/60))
# result = pd.read_csv('%s/result/ver6_lr_stacking.csv' % data_path)
# result.to_csv('ver6_lr_stacking.csv', index = False)
# print('%0.2f min: Finish loading result'%((time.time() - start_time)/60))

# %0.2f は小数点2位まで表示
print('%0.2f min: Start loading data'%((time.time() - start_time)/60))
sale_train = pd.read_csv('%s/sales_train.csv' % data_path)
test  = pd.read_csv('%s/test.csv' % data_path)

sale_train[sale_train['item_id'] == 11373][['item_price']].sort_values(['item_price'])
sale_train[sale_train['item_id'] == 11365].sort_values(['item_price'])

# Correct sale_train values
sale_train['item_price'][2909818] = np.nan
sale_train['item_cnt_day'][2909818] = np.nan
sale_train['item_price'][2909818] = sale_train[(sale_train['shop_id'] ==12) & (sale_train['item_id'] == 11373) & (sale_train['date_block_num'] == 33)]['item_price'].median()
sale_train['item_cnt_day'][2909818] = round(sale_train[(sale_train['shop_id'] ==12) & (sale_train['item_id'] == 11373) & (sale_train['date_block_num'] == 33)]['item_cnt_day'].median())
sale_train['item_price'][885138] = np.nan
sale_train['item_price'][885138] = sale_train[(sale_train['item_id'] == 11365) & (sale_train['shop_id'] ==12) & (sale_train['date_block_num'] == 8)]['item_price'].median()

test_nrow = test.shape[0]
sale_train = sale_train.merge(test[['shop_id']].drop_duplicates(), how = 'inner')
sale_train['date'] = pd.to_datetime(sale_train['date'], format = '%d.%m.%Y')
print('%0.2f min: Finish loading data'%((time.time() - start_time)/60))



# 1. Aggregate data ----------------------------------------------------------------

from itertools import product

# 月、店舗、商品３軸でgridを作る
grid = []
for block_num in sale_train['date_block_num'].unique():
    cur_shops = sale_train[sale_train['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sale_train[sale_train['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

# gridをdfに変換する
index_cols = ['shop_id', 'item_id', 'date_block_num']
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
print('%0.2f min: Finish creating the grid'%((time.time() - start_time)/60))

index_cols = ['shop_id', 'item_id', 'date_block_num']
# item_cnt_dayの閾値を0~20とする
sale_train['item_cnt_day'] = sale_train['item_cnt_day'].clip(0,20)
gb_cnt = sale_train.groupby(index_cols)['item_cnt_day'].agg(['sum']).reset_index().rename(columns = {'sum': 'item_cnt_month'})
gb_cnt['item_cnt_month'] = gb_cnt['item_cnt_month'].clip(0,20).astype(np.int)

# 空のdfと集計後の数字をマージ
train = pd.merge(grid,gb_cnt,how='left',on=index_cols).fillna(0)
train['item_cnt_month'] = train['item_cnt_month'].astype(int)
train = downcast_dtypes(train)
train.sort_values(['date_block_num','shop_id','item_id'],inplace=True)
print('%0.2f min: Finish joining gb_cnt'%((time.time() - start_time)/60))

```

##　参考資料
https://elitedatascience.com/feature-engineering　　
