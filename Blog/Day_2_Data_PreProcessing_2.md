# 2日目データ事前処理-2　Day 2 Data PreProcessing - 2

本日の目標は
1. カテゴリデータのコード化
2. データセットの分割
3. 説明変数のスケーリング

## Step 1: カテゴリデータのコード化
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# Xの１列目をコード化し、本列を置換する。
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
# 下記同意
# labelencoder_X.fit(X[ : , 0 ])
# X[ : , 0] = labelencoder_X.transform(X[ : , 0])

# ダミー変数作成
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Yをコード化し、本列を置換する。
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
```

## Step 2: データセットの分割
```python
from sklearn.cross_validation import train_test_split

# XとYを80-20でtraining set と　test set 分割
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
```

## Step 3: 説明変数のスケーリング
```python
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Xのtraining set と　test set をfeature Scaling 処理
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
```
## 補足
### Numpy OneHotEncoder と　Pandas.get_dummies の違い

> OneHotEncoder cannot process string values directly. If your nominal features are strings, then you need to first map them into integers.
> pandas.get_dummies is kind of the opposite. By default, it only converts string columns into one-hot representation, unless columns are specified.

OneHotEncoderのであれば、文字列をコード化してから、変換する
```python
 labelencoder_X.fit(X[ : , 0 ])
 X[ : , 0] = labelencoder_X.transform(X[ : , 0])
 onehotencoder = OneHotEncoder(categorical_features = [0])
 X = onehotencoder.fit_transform(X).toarray()
```
get_dummiesであれば、文字列のまま変換する。（コードが変換できない）
```python
df = pd.get_dummies(df, columns = X[ : , 0])
```
### Feature Scalingの目的
説明変数の範囲のばらつきを、全ての説明変数の値の範囲を、同じ範囲とする。
よく使う手法は：
1. xi=xi−mean(x) / sd(x)
2. xi=xi−mean(x) / max(x)−min(x)

### 参考資料
https://qiita.com/kibinag0/items/723f95277263921650b4
https://stackoverflow.com/questions/36631163/pandas-get-dummies-vs-sklearns-onehotencoder-what-is-more-efficient
http://docs.pyq.jp/python/machine_learning/tips/train_test_split.html
https://qiita.com/katsu1110/items/6a6d5c5a6b8af8fbf813
