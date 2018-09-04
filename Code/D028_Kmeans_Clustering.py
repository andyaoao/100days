import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# ランダムでデータを生成する
dataset = datasets.make_blobs()

# featureのみ残し、ラベルを外す（2軸のデータを生成された）
features = dataset[0]
# targets = dataset[1]

# クラスタ数（何分類まで分類されると定義する）
N_CLUSTERS = 5

# クラスタリングモデルを生成する
cls = KMeans(n_clusters=N_CLUSTERS)

# 予測する
pred = cls.fit_predict(features)

# 各分類をラベルごとに色付けして表示する
for i in range(N_CLUSTERS):
    labels = features[pred == i]
    plt.scatter(labels[:, 0], labels[:, 1])

# クラスタのcentralを他の色で描く
centers = cls.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=100,
            facecolors='none', edgecolors='black')

plt.show()

#
# from matplotlib import pyplot as plt
# from sklearn import datasets
# import numpy as np
#
#
# class KMeans(object):
#     """KMeans 法でクラスタリングするクラス"""
#
#     def __init__(self, n_clusters=2, max_iter=300):
#         """コンストラクタ
#
#         Args:
#             n_clusters (int): クラスタ数
#             max_iter (int): 最大イテレーション数
#         """
#         self.n_clusters = n_clusters
#         self.max_iter = max_iter
#
#         self.cluster_centers_ = None
#
#     def fit_predict(self, features):
#         """クラスタリングを実施する
#
#         Args:
#             features (numpy.ndarray): ラベル付けするデータ
#
#         Returns:
#             numpy.ndarray: ラベルデータ
#         """
#         # 要素の中からセントロイド (重心) の初期値となる候補をクラスタ数だけ選び出す
#         feature_indexes = np.arange(len(features))
#         np.random.shuffle(feature_indexes)
#         initial_centroid_indexes = feature_indexes[:self.n_clusters]
#         self.cluster_centers_ = features[initial_centroid_indexes]
#
#         # ラベル付けした結果となる配列はゼロで初期化しておく
#         pred = np.zeros(features.shape)
#
#         # クラスタリングをアップデートする
#         for _ in range(self.max_iter):
#             # 各要素から最短距離のセントロイドを基準にラベルを更新する
#             new_pred = np.array([
#                 np.array([
#                     self._euclidean_distance(p, centroid)
#                     for centroid in self.cluster_centers_
#                 ]).argmin()
#                 for p in features
#             ])
#
#             if np.all(new_pred == pred):
#                 # 更新前と内容が同じなら終了
#                 break
#
#             pred = new_pred
#
#             # 各クラスタごとにセントロイド (重心) を再計算する
#             self.cluster_centers_ = np.array([features[pred == i].mean(axis=0)
#                                               for i in range(self.n_clusters)])
#
#         return pred
#
#     def _euclidean_distance(self, p0, p1):
#         return np.sum((p0 - p1) ** 2)
#
#
# def main():
#     # クラスタ数
#     N_CLUSTERS = 5
#
#     # Blob データを生成する
#     dataset = datasets.make_blobs(centers=N_CLUSTERS)
#
#     # 特徴データ
#     features = dataset[0]
#     # 正解ラベルは使わない
#     # targets = dataset[1]
#
#     # クラスタリングする
#     cls = KMeans(n_clusters=N_CLUSTERS)
#     pred = cls.fit_predict(features)
#
#     # 各要素をラベルごとに色付けして表示する
#     for i in range(N_CLUSTERS):
#         labels = features[pred == i]
#         plt.scatter(labels[:, 0], labels[:, 1])
#
#     centers = cls.cluster_centers_
#     plt.scatter(centers[:, 0], centers[:, 1], s=100,
#                 facecolors='none', edgecolors='black')
#
#     plt.show()
