# 74日目 Encoding 1 Day74 Encoding 1

本日の目標は
1. Encodingの方法

## Step 1: Encodingの方法

1. label encoding
単純にカテゴリの種類を0-mまで、codingする。  

```python
for col in cat_cols:
    le = LabelEncoder()
    le.fit(np.concatenate([train[col], test[col]]))
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
```

2. one hot encoding
各種類のカテゴリを一つのcolumnとして存在し、valueは0か1のみとする。  
カテゴリの種類が多すぎる場合、相応しくないと思われる。  

```python
df = train.append(test).reset_index()
original_column = list(df.columns)
df = pd.get_dummies(df, columns = cat_cols, dummy_na = True)
new_column = [c for c in df.columns if c not in original_column ]
```

3. target encoding (mean encoding, likelihood encoding, impact encoding)
予測のtarget数 / この種類のカテゴリの数　でtarget encodingを算出する。  

```python
# ２段階cross Validationで、外部20folds,内部10foldsを設定する
n_folds = 20
n_inner_folds = 10
likelihood_encoded = pd.Series()
likelihood_coding_map = {}

# 全体mean
oof_default_mean = train[target].mean()
kf = KFold(n_splits=n_folds, shuffle=True)
oof_mean_cv = pd.DataFrame()
split = 0

# ２段階cross Validation
for infold, oof in kf.split(train[feature]):
    print ('==============level 1 encoding..., fold %s ============' % split)
    inner_kf = KFold(n_splits=n_inner_folds, shuffle=True)
    # inner_foldのmeanを計算
    inner_oof_default_mean = train.iloc[infold][target].mean()
    inner_split = 0
    inner_oof_mean_cv = pd.DataFrame()
    likelihood_encoded_cv = pd.Series()
    for inner_infold, inner_oof in inner_kf.split(train.iloc[infold]):
        print ('==============level 2 encoding..., inner fold %s ============' % inner_split)
        # 内部fold内のmean
        oof_mean = train.iloc[inner_infold].groupby(by=feature)[target].mean()
        # 外部assign oof_mean to the infold
        likelihood_encoded_cv = likelihood_encoded_cv.append(train.iloc[infold].apply(
            lambda x : oof_mean[x[feature]]
            if x[feature] in oof_mean.index
            else inner_oof_default_mean, axis = 1))
        inner_oof_mean_cv = inner_oof_mean_cv.join(pd.DataFrame(oof_mean), rsuffix=inner_split, how='outer')
        inner_oof_mean_cv.fillna(inner_oof_default_mean, inplace=True)
        inner_split += 1

    oof_mean_cv = oof_mean_cv.join(pd.DataFrame(inner_oof_mean_cv), rsuffix=split, how='outer')
    oof_mean_cv.fillna(value=oof_default_mean, inplace=True)
    split += 1
    print ('============final mapping...===========')
    likelihood_encoded = likelihood_encoded.append(train.iloc[oof].apply(
        lambda x: np.mean(inner_oof_mean_cv.loc[x[feature]].values)
        if x[feature] in inner_oof_mean_cv.index
        else oof_default_mean, axis=1))

######################################### map into test dataframe
train[feature] = likelihood_encoded
likelihood_coding_mapping = oof_mean_cv.mean(axis = 1)
default_coding = oof_default_mean

likelihood_coding_map[feature] = (likelihood_coding_mapping, default_coding)
mapping, default_mean = likelihood_coding_map[feature]
test[feature] = test.apply(lambda x : mapping[x[feature]]
                                       if x[feature] in mapping
                                       else default_mean,axis = 1)
```

4. beta target encoding
```python
# Step 1: fill NA in train and test dataframe

# Step 2: 5-fold CV (beta target encoding within each fold)

kf = KFold(n_splits=5, shuffle=True, random_state=0)
for i, (dev_index, val_index) in enumerate(kf.split(train.index.values)):
    # split data into dev set and validation set
    dev = train.loc[dev_index].reset_index(drop=True)
    val = train.loc[val_index].reset_index(drop=True)

    feature_cols = []    
    for var_name in cat_cols:
        feature_name = f'{var_name}_mean'
        feature_cols.append(feature_name)

        prior_mean = np.mean(dev[target_col])
        stats = dev[[target_col, var_name]].groupby(var_name).agg(['sum', 'count'])[target_col].reset_index()           

        ### beta target encoding by Bayesian average for dev set
        df_stats = pd.merge(dev[[var_name]], stats, how='left')
        df_stats['sum'].fillna(value = prior_mean, inplace = True)
        df_stats['count'].fillna(value = 1.0, inplace = True)
        N_prior = np.maximum(N_min - df_stats['count'].values, 0)   # prior parameters
        dev[feature_name] = (prior_mean * N_prior + df_stats['sum']) / (N_prior + df_stats['count']) # Bayesian mean

        ### beta target encoding by Bayesian average for val set
        df_stats = pd.merge(val[[var_name]], stats, how='left')
        df_stats['sum'].fillna(value = prior_mean, inplace = True)
        df_stats['count'].fillna(value = 1.0, inplace = True)
        N_prior = np.maximum(N_min - df_stats['count'].values, 0)   # prior parameters
        val[feature_name] = (prior_mean * N_prior + df_stats['sum']) / (N_prior + df_stats['count']) # Bayesian mean

        ### beta target encoding by Bayesian average for test set
        df_stats = pd.merge(test[[var_name]], stats, how='left')
        df_stats['sum'].fillna(value = prior_mean, inplace = True)
        df_stats['count'].fillna(value = 1.0, inplace = True)
        N_prior = np.maximum(N_min - df_stats['count'].values, 0)   # prior parameters
        test[feature_name] = (prior_mean * N_prior + df_stats['sum']) / (N_prior + df_stats['count']) # Bayesian mean

        # Bayesian mean is equivalent to adding N_prior data points of value prior_mean to the data set.
        del df_stats, stats
    # Step 3: train model (K-fold CV), get oof prediction
```

##　参考資料
https://medium.com/datadriveninvestor/improve-your-classification-models-using-mean-target-encoding-a3d573df31e8  　
https://zhuanlan.zhihu.com/p/40231966  
