# 70日目 Feature Engineering 1 Day70 Feature Engineering 1

本日の目標は
1. Mean encoding の実装

## Step 1: Mean encoding の実装
```python
# 2.4 Scale feature columns --------------------------------------------------------

from sklearn.preprocessing import StandardScaler
# 時間帯区分の最後のみtest set
train = all_data[all_data['date_block_num']!= all_data['date_block_num'].max()]
test = all_data[all_data['date_block_num']== all_data['date_block_num'].max()]
sc = StandardScaler()

to_drop_cols = ['date_block_num']
feature_columns = list(set(lag_cols + index_cols + list(date_columns)).difference(to_drop_cols))

print ("feature_columns")
print (feature_columns)

train[feature_columns] = sc.fit_transform(train[feature_columns])
test[feature_columns] = sc.transform(test[feature_columns])
all_data = pd.concat([train, test], axis = 0)
all_data = downcast_dtypes(all_data)

del train, test, date_features, sale_train
gc.collect()

print('%0.2f min: Finish scaling features'%((time.time() - start_time)/60))

# 3. First-level model ------------------------------------------------------------------

# Save `date_block_num`, as we can't use them as features, but will need them to split the dataset into parts

dates = all_data['date_block_num']
last_block = dates.max()

print('Test `date_block_num` is %d' % last_block)
print(feature_columns)

print('%0.2f min: Start training First level models'%((time.time() - start_time)/60))

start_first_level_total = time.perf_counter()
scoringMethod = 'r2'; from sklearn.metrics import mean_squared_error; from math import sqrt

# Train meta-features M = 15 (12 + 15 = 27)

months_to_generate_meta_features = range(27,last_block +1)
mask = dates.isin(months_to_generate_meta_features)
Target = 'item_cnt_month'
y_all_level2 = all_data[Target][mask].values
X_all_level2 = np.zeros([y_all_level2.shape[0], num_first_level_models])

# Now fill `X_train_level2` with metafeatures

slice_start = 0


```

##　参考資料
https://qiita.com/SS1031/items/38514e0fb1f43b101748  　
