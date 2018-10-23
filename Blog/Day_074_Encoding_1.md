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
n_folds = 20
n_inner_folds = 10
likelihood_encoded = pd.Series()
likelihood_coding_map = {}

oof_default_mean = train[target].mean()      # global prior mean
kf = KFold(n_splits=n_folds, shuffle=True)
oof_mean_cv = pd.DataFrame()
split = 0

for infold, oof in kf.split(train[feature]):
    print ('==============level 1 encoding..., fold %s ============' % split)
    inner_kf = KFold(n_splits=n_inner_folds, shuffle=True)
    inner_oof_default_mean = train.iloc[infold][target].mean()
    inner_split = 0
    inner_oof_mean_cv = pd.DataFrame()

    likelihood_encoded_cv = pd.Series()
    for inner_infold, inner_oof in inner_kf.split(train.iloc[infold]):
        print ('==============level 2 encoding..., inner fold %s ============' % inner_split)
        # inner out of fold mean
        oof_mean = train.iloc[inner_infold].groupby(by=feature)[target].mean()
        # assign oof_mean to the infold
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
5. 不做处理（模型自动编码）


##　参考資料
https://medium.com/datadriveninvestor/improve-your-classification-models-using-mean-target-encoding-a3d573df31e8  　
https://zhuanlan.zhihu.com/p/40231966  
