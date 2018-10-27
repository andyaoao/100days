# 70日目 Feature Engineering 1 Day70 Feature Engineering 1

本日の目標は
1. Mean encoding の実装

## Step 1: Mean encoding の実装
```python
# 2.1 Combine trainset and testset -----------------------------------------

print('%0.2f min: Start combining data'%((time.time() - start_time)/60))
# validation使用するかどうか
if Validation == False:
    # testデータに時間帯区分を追加
    test['date_block_num'] = 34
    all_data = pd.concat([train, test], axis = 0)
    all_data = all_data.drop(columns = ['ID'])

else:
    # testデータを使わない
    all_data = train

del train, test, col_tr
gc.collect()

all_data = downcast_dtypes(all_data)

# 2.2 Creating item/shop pair lags lag-based features ----------------------------

print('%0.2f min: Start adding lag-based feature'%((time.time() - start_time)/60))

index_cols = ['shop_id', 'item_id', 'item_category_id', 'item_cat_id_fix', 'date_block_num']
cols_to_rename = list(all_data.columns.difference(index_cols))
print(cols_to_rename)

# lag periodにより、新しいfeatureを作る
shift_range = [1, 2, 3, 4, 12]
for month_shift in tqdm(shift_range):
    train_shift = all_data[index_cols + cols_to_rename].copy()
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift

    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)
    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

del train_shift
gc.collect()

all_data = all_data[all_data['date_block_num'] >= 12] # Don't use old data from year 2013
lag_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]]
all_data = downcast_dtypes(all_data)
print('%0.2f min: Finish generating lag features'%((time.time() - start_time)/60))


# 2.3 Creating date features --------------------------------------------------------

print('%0.2f min: Start getting date features'%((time.time() - start_time)/60))

# 日付に対して整理
dates_train = sale_train[['date', 'date_block_num']].drop_duplicates()
# 予測対象の一年前の日付をテスト対象とする
dates_test = dates_train[dates_train['date_block_num'] == 34-12]
dates_test['date_block_num'] = 34
dates_test['date'] = dates_test['date'] + pd.DateOffset(years=1)
dates_all = pd.concat([dates_train, dates_test])

# 日付をパラしてfeatureを作る
dates_all['dow'] = dates_all['date'].dt.dayofweek
dates_all['year'] = dates_all['date'].dt.year
dates_all['month'] = dates_all['date'].dt.month
dates_all = pd.get_dummies(dates_all, columns=['dow'])
dow_col = ['dow_' + str(x) for x in range(7)]
date_features = dates_all.groupby(['year', 'month', 'date_block_num'])[dow_col].agg('sum').reset_index()
date_features['days_of_month'] = date_features[dow_col].sum(axis=1)
date_features['year'] = date_features['year'] - 2013

date_features = date_features[['month', 'year', 'days_of_month', 'date_block_num']]
# 整理した後のdate featureをトランdataとmerge
all_data = all_data.merge(date_features, on = 'date_block_num', how = 'left')
date_columns = date_features.columns.difference(set(index_cols))
print('%0.2f min: Finish getting date features'%((time.time() - start_time)/60))


```

##　参考資料
https://qiita.com/SS1031/items/38514e0fb1f43b101748  　
