# 32日目 Principal Component Analysis 2 Day32 Principal Component Analysis 2

本日の目標は
1. PCAの次元圧縮ロジック理解
2. 次元圧縮をを自分で実装してみる

## Step 1: PCAの次元圧縮ロジック理解
次元(feature)間の共分散(covariance)マトリックスを作成する。相関(correlation)マトリックスを作る場合もある。  
固有ベクター(Eigenvectors)と固有値(Eigenvalues)を計算する。  
固有ベクターは方向を決める；固有値は解説能力を決める。  
固有値の高い順で、欲しい次元数を選択する。  
元のサンプルデータを選択された固有ベクターに投影する。  

## Step 2: 次元圧縮をを自分で実装してみる
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ランダムでデータ10個を生成する(4featureがある)
dataset = datasets.make_blobs(n_samples=10,n_features=4)

# featureのみ残し、ラベルを外す（2軸のデータを生成された）
X = np.array(dataset[0])

# PCA手法に、feature scalingは必要
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 共分散のマトリックスを作成
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)
# numpyでconvarianceを一気に計算する方法
# print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

# 固定ベクターと固定値の計算
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# 固定ベクターと固定値のペアを作成し、高い順を作成する
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort()
eig_pairs.reverse()

# 解説能力を確認する。全体を100として、上位から順番に累計したら、どこまで選択すべき(主成分の次元数を決める手法)
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print (cum_var_exp)

# ４つの中、最上位の２個を選択し、次元削減用の計算マトリックスを作成
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))

print('Matrix W:\n', matrix_w)

# 元のサンプルデータを選択された固有ベクターに投影する
Y = X_std.dot(matrix_w)
```

## 補足

### 参考資料
PCAの実装 https://plot.ly/ipython-notebooks/principal-component-analysis/  
固有ベクターの解説  
https://postd.cc/a-beginners-guide-to-eigenvectors-pca-covariance-and-entropy/#principal  
https://deepage.net/features/numpy-eigenvalue-vector.html  
