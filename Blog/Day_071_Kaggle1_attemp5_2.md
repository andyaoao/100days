# 70日目 Feature Engineering 1 Day70 Feature Engineering 1

本日の目標は
1. Two-level model 実装してみる


## Step 1: Feature Engineeringの簡単概念
```python
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

# item categoryをtrainとtestのdata set に入れる
item = pd.read_csv('%s/items.csv' % data_path)
train = train.merge(item[['item_id', 'item_category_id']], on = ['item_id'], how = 'left')
test = test.merge(item[['item_id', 'item_category_id']], on = ['item_id'], how = 'left')
print('%0.2f min: Finish adding item_category_id'%((time.time() - start_time)/60))

# item categoryに対して、再分類・採番
item_cat = pd.read_csv('%s/item_categories.csv' % data_path)
l_cat = list(item_cat.item_category_name)
for ind in range(0,1):
    l_cat[ind] = 'PC Headsets / Headphones'
for ind in range(1,8):
    l_cat[ind] = 'Access'
l_cat[8] = 'Tickets (figure)'
l_cat[9] = 'Delivery of goods'
for ind in range(10,18):
    l_cat[ind] = 'Consoles'
for ind in range(18,25):
    l_cat[ind] = 'Consoles Games'
l_cat[25] = 'Accessories for games'
for ind in range(26,28):
    l_cat[ind] = 'phone games'
for ind in range(28,32):
    l_cat[ind] = 'CD games'
for ind in range(32,37):
    l_cat[ind] = 'Card'
for ind in range(37,43):
    l_cat[ind] = 'Movie'
for ind in range(43,55):
    l_cat[ind] = 'Books'
for ind in range(55,61):
    l_cat[ind] = 'Music'
for ind in range(61,73):
    l_cat[ind] = 'Gifts'
for ind in range(73,79):
    l_cat[ind] = 'Soft'
for ind in range(79,81):
    l_cat[ind] = 'Office'
for ind in range(81,83):
    l_cat[ind] = 'Clean'
l_cat[83] = 'Elements of a food'

# item category の新コードをtrain test set に追加
from sklearn import preprocessing
lb = preprocessing.LabelEncoder()
item_cat['item_cat_id_fix'] = lb.fit_transform(l_cat)
train = train.merge(item_cat[['item_cat_id_fix', 'item_category_id']], on = ['item_category_id'], how = 'left')
test = test.merge(item_cat[['item_cat_id_fix', 'item_category_id']], on = ['item_category_id'], how = 'left')

del item, item_cat, grid, gb_cnt
gc.collect()
print('%0.2f min: Finish adding item_cat_id_fix'%((time.time() - start_time)/60))

# 2. Add item/shop pair mean-encodings -----------------------------------------

# For Trainset
print('%0.2f min: Start adding mean-encoding for item_cnt_month'%((time.time() - start_time)/60))
Target = 'item_cnt_month'
# item_cnt_itemの全体の平均値を計算する
global_mean =  train[Target].mean()
y_tr = train[Target].values

mean_encoded_col = ['shop_id', 'item_id', 'item_category_id', 'item_cat_id_fix']
for col in tqdm(mean_encoded_col):
    # 4種類のidを分けて、idとint_count_monthを抽出
    col_tr = train[[col] + [Target]]
    corrcoefs = pd.DataFrame(columns = ['Cor'])

    # クロスバリデーション (K-fold)
    from sklearn.model_selection import KFold

    kf = KFold(n_splits = 5, shuffle = False, random_state = SEED)
    col_tr[col + '_cnt_month_mean_Kfold'] = global_mean

    for tr_ind, val_ind in kf.split(col_tr):
        X_tr, X_val = col_tr.iloc[tr_ind], col_tr.iloc[val_ind]
        means = X_val[col].map(X_tr.groupby(col)[Target].mean())
        X_val[col + '_cnt_month_mean_Kfold'] = means
        col_tr.iloc[val_ind] = X_val
        print (X_val.head())

    col_tr.fillna(global_mean, inplace = True)
    corrcoefs.loc[col + '_cnt_month_mean_Kfold'] = np.corrcoef(y_tr, col_tr[col + '_cnt_month_mean_Kfold'])[0][1]
```

##　参考資料
https://elitedatascience.com/feature-engineering　　
