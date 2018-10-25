# 70日目 Feature Engineering 1 Day70 Feature Engineering 1

本日の目標は
1. Mean encoding の実装

## Step 1: Mean encoding の実装
```python
# 2. Add item/shop pair mean-encodings -----------------------------------------

print('%0.2f min: Start adding mean-encoding for item_cnt_month'%((time.time() - start_time)/60))
Target = 'item_cnt_month'
# item_cnt_itemの全体の平均値を計算する
global_mean =  train[Target].mean()
y_tr = train[Target].values

mean_encoded_col = ['shop_id', 'item_id', 'item_category_id', 'item_cat_id_fix']
for col in tqdm(mean_encoded_col):
    # 4種類のidを分けて、id(shop_id, item_id, item_category_id, item_cat_id_fix)ごとのint_count_monthを抽出
    col_tr = train[[col] + [Target]]
    corrcoefs = pd.DataFrame(columns = ['Cor'])
    # print ("col_tr")
    # print (col_tr)

    # クロスバリデーション (K-fold)
    from sklearn.model_selection import KFold

    # training set を 5 splitを分ける
    kf = KFold(n_splits = 5, shuffle = False, random_state = SEED)
    col_tr[col + '_cnt_month_mean_Kfold'] = global_mean

    # 4種類のidごとのint_count_monthのid単位のmeanを全レコードの横に持たせる
    for tr_ind, val_ind in kf.split(col_tr):
        X_tr, X_val = col_tr.iloc[tr_ind], col_tr.iloc[val_ind]
        # id単位のmeanを計算する（shop_idごと、item_idごと）
        # training setのmeanをvalidation setに適用
        means = X_val[col].map(X_tr.groupby(col)[Target].mean())
        # print ("means")
        # print (means)
        X_val[col + '_cnt_month_mean_Kfold'] = means
        col_tr.iloc[val_ind] = X_val
        # print("X_val")
        # print (X_val.head())

    col_tr.fillna(global_mean, inplace = True)
    # 店舗商品ごとの月別買い上げ点数とidのmeanの相関係数
    corrcoefs.loc[col + '_cnt_month_mean_Kfold'] = np.corrcoef(y_tr, col_tr[col + '_cnt_month_mean_Kfold'])[0][1]

    # idごとの点数合計、カウントを計算
    item_id_target_sum = col_tr.groupby(col)[Target].sum()
    item_id_target_count = col_tr.groupby(col)[Target].count()
    col_tr[col + '_cnt_month_sum'] = col_tr[col].map(item_id_target_sum)
    col_tr[col + '_cnt_month_count'] = col_tr[col].map(item_id_target_count)
    col_tr[col + '_target_mean_LOO'] = (col_tr[col + '_cnt_month_sum'] - col_tr[Target]) / (col_tr[col + '_cnt_month_count'] - 1)
    col_tr.fillna(global_mean, inplace = True)
    corrcoefs.loc[col + '_target_mean_LOO'] = np.corrcoef(y_tr, col_tr[col + '_target_mean_LOO'])[0][1]
    print ("col_tr")
    print (col_tr)

    # smoothing
    item_id_target_mean = col_tr.groupby(col)[Target].mean()
    item_id_target_count = col_tr.groupby(col)[Target].count()
    col_tr[col + '_cnt_month_mean'] = col_tr[col].map(item_id_target_mean)
    col_tr[col + '_cnt_month_count'] = col_tr[col].map(item_id_target_count)
    alpha = 100
    # smoothing の方法はよくわからない
    col_tr[col + '_cnt_month_mean_Smooth'] = (col_tr[col + '_cnt_month_mean'] *  col_tr[col + '_cnt_month_count'] + global_mean * alpha) / (alpha + col_tr[col + '_cnt_month_count'])
    col_tr[col + '_cnt_month_mean_Smooth'].fillna(global_mean, inplace=True)
    corrcoefs.loc[col + '_cnt_month_mean_Smooth'] = np.corrcoef(y_tr, col_tr[col + '_cnt_month_mean_Smooth'])[0][1]

    # target encodingの実装
    cumsum = col_tr.groupby(col)[Target].cumsum() - col_tr[Target]
    sumcnt = col_tr.groupby(col).cumcount()
    col_tr[col + '_cnt_month_mean_Expanding'] = cumsum / sumcnt
    col_tr[col + '_cnt_month_mean_Expanding'].fillna(global_mean, inplace=True)
    corrcoefs.loc[col + '_cnt_month_mean_Expanding'] = np.corrcoef(y_tr, col_tr[col + '_cnt_month_mean_Expanding'])[0][1]

    train = pd.concat([train, col_tr[corrcoefs['Cor'].idxmax()]], axis = 1)
    print(corrcoefs.sort_values('Cor'))
    print('%0.2f min: Finish encoding %s'%((time.time() - start_time)/60, col))

print('%0.2f min: Finish adding mean-encoding'%((time.time() - start_time)/60))

```

##　参考資料
https://qiita.com/SS1031/items/38514e0fb1f43b101748  　
