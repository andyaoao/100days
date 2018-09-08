import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ランダムでデータ10個を生成する
dataset = datasets.make_blobs(n_samples=10)

# featureのみ残し、ラベルを外す（2軸のデータを生成された）
X = np.array(dataset[0])

# PCA手法に、feature scalingは必要
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# PCAの実装
pca = PCA(n_components=1, svd_solver="full")
# PCAを通して、次元圧縮した状態
X_project = pca.fit_transform(X_norm)
# 次元圧縮されたデータを、次元復元
X_recover = pca.inverse_transform(X_project)
# 元データの１個目
print("Original fist sample:", X_norm[0, :])
# PCAで次元圧縮後
print("Projection of the first example:", X_project[0])
# 削減したデータの次元を戻す
print("Approximation of the first example:", X_recover[0,:])
# 説明される分散比（n_componentsはこれで調整する、次元削減しない[=N]なら1）
print("Explained variance ratio:", pca.explained_variance_ratio_)
print()
