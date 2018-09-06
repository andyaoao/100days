from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np

# ランダムでデータ10個を生成する
dataset = datasets.make_blobs(n_samples=10)

# featureのみ残し、ラベルを外す（2軸のデータを生成された）
X = dataset[0]

# 生成したデータをラベルつける
labels = range(1, 11)
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')

for label, x, y in zip(labels, X[:, 0], X[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-3, 3),
        textcoords='offset points', ha='right', va='bottom')
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage

# linkage functionを使って、サンプルポイント間の距離を計算した後、dendrogramで出力できる形に変換
linked = linkage(X, 'single')

labelList = range(1, 11)

plt.figure(figsize=(10, 7))

# dendrogram出力のfunction
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()

from sklearn.cluster import AgglomerativeClustering

# AgglomerativeClusteringでモデルを作成
# n_clustersはクラスターの数；affinityは最小距離の計算方法；linkageはクラスター間の関連性の計算方法
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(X)

plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()
