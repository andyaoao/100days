# 79日目 Feature Engineering 1 Day79 Feature Engineering 1

本日の目標は
1. Mean encoding の実装

## Step 1: Mean encoding の実装
```python
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


for cur_block_num in tqdm(months_to_generate_meta_features):

    print('-' * 50)
    print('Start training for month%d'% cur_block_num)

    start_cur_month = time.perf_counter()
    # 対象時間帯より早いのはtrain setに入れる、対象はtestに入れる
    cur_X_train = all_data.loc[dates <  cur_block_num][feature_columns]
    cur_X_test =  all_data.loc[dates == cur_block_num][feature_columns]
    cur_y_train = all_data.loc[dates <  cur_block_num, Target].values
    cur_y_test =  all_data.loc[dates == cur_block_num, Target].values

    # Create Numpy arrays of train, test and target dataframes to feed into models

    train_x = cur_X_train.values
    train_y = cur_y_train.ravel()
    test_x = cur_X_test.values
    test_y = cur_y_test.ravel()

    preds = []

    from sklearn.linear_model import (LinearRegression, SGDRegressor)
    import lightgbm as lgb

    sgdr= SGDRegressor(
        penalty = 'l2' ,
        random_state = SEED )
    lgb_params = {
                  'feature_fraction': 0.75,
                  'metric': 'rmse',
                  'nthread':1,
                  'min_data_in_leaf': 2**7,
                  'bagging_fraction': 0.75,
                  'learning_rate': 0.03,
                  'objective': 'mse',
                  'bagging_seed': 2**7,
                  'num_leaves': 2**7,
                  'bagging_freq':1,
                  'verbose':0
                  }

    estimators = [sgdr]

    for estimator in estimators:
        print('Training Model %d: %s'%(len(preds), estimator.__class__.__name__))

        start = time.perf_counter()
        estimator.fit(train_x, train_y)
        pred_test = estimator.predict(test_x)
        preds.append(pred_test)

        # pred_train = estimator.predict(train_x)
        # print('Train RMSE for %s is %f' % (estimator.__class__.__name__, sqrt(mean_squared_error(cur_y_train, pred_train))))
        # print('Test RMSE for %s is %f' % (estimator.__class__.__name__, sqrt(mean_squared_error(cur_y_test, pred_test))))

        run = time.perf_counter() - start

        print('{} runs for {:.2f} seconds.'.format(estimator.__class__.__name__, run))
        print()

    print('Training Model %d: %s'%(len(preds), 'lightgbm'))

    start = time.perf_counter()

    estimator = lgb.train(lgb_params, lgb.Dataset(train_x, label=train_y), 300)
    pred_test = estimator.predict(test_x)
    preds.append(pred_test)

    # pred_train = estimator.predict(train_x)
    # print('Train RMSE for %s is %f' % ('lightgbm', sqrt(mean_squared_error(cur_y_train, pred_train))))
    # print('Test RMSE for %s is %f' % ('lightgbm', sqrt(mean_squared_error(cur_y_test, pred_test))))

    run = time.perf_counter() - start
    print('{} runs for {:.2f} seconds.'.format('lightgbm', run))

    print()

    print('Training Model %d: %s'%(len(preds), 'keras'))

    start = time.perf_counter()

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor

    def baseline_model():
    	# create model
        model = Sequential()
        model.add(Dense(20, input_dim=train_x.shape[1], kernel_initializer='uniform', activation='softplus'))
        model.add(Dense(1, kernel_initializer='uniform', activation = 'relu'))
        # Compile model
        model.compile(loss='mse', optimizer='Nadam', metrics=['mse'])
        # model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    estimator = KerasRegressor(build_fn=baseline_model, verbose=1, epochs=5, batch_size = 55000)

    estimator.fit(train_x, train_y)
    pred_test = estimator.predict(test_x)
    preds.append(pred_test)

    run = time.perf_counter() - start
    print('{} runs for {:.2f} seconds.'.format('lightgbm', run))

    cur_month_run_total = time.perf_counter() - start_cur_month

    print('Total running time was {:.2f} minutes.'.format(cur_month_run_total/60))
    print('-' * 50)

    slice_end = slice_start + cur_X_test.shape[0]
    X_all_level2[ slice_start : slice_end , :] = np.c_[preds].transpose()
    slice_start = slice_end

# Split train and test

test_nrow = len(preds[0])
X_train_level2 = X_all_level2[ : -test_nrow, :]
X_test_level2 = X_all_level2[ -test_nrow: , :]
y_train_level2 = y_all_level2[ : -test_nrow]
y_test_level2 = y_all_level2[ -test_nrow : ]

print('%0.2f min: Finish training First level models'%((time.perf_counter() - start_first_level_total)/60))


```

##　参考資料
https://kknews.cc/news/rpyxvv.html  
