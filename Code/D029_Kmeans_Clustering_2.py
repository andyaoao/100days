from matplotlib import pyplot as plt
from sklearn import datasets
import numpy as np


# KMeans clusteringを自分で実装してみる
class KMeans(object):

    # クラスを定義する
    # クラスター数とiterationの回数を定義する
    def __init__(self, n_clusters=2, max_iter=300):

        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self.cluster_centers_ = None

    def fit_predict(self, features):

        # featureの中より、ランダムで中心を選択する(個数は、initで定義されたクラスタ数)
        feature_indexes = np.arange(len(features))
        np.random.shuffle(feature_indexes)
        initial_centroid_indexes = feature_indexes[:self.n_clusters]
        self.cluster_centers_ = features[initial_centroid_indexes]

        # 予測用のarrayは一回zeroで初期化する
        pred = np.zeros(features.shape)

        # クラスタリングをアップデートする
        for _ in range(self.max_iter):
            # pはfeatureの中の各point、centroidは中心点
            # 一つのポイントは各クラスターの中心とそれぞれ計算する。
            # 計算結果を比較して、最小値の方は、このポイントの所属クラスター。
            new_pred = np.array([
                np.array([
                    self._euclidean_distance(p, centroid)
                    for centroid in self.cluster_centers_
                ]).argmin()
                for p in features
            ])

            if np.all(new_pred == pred):
                # 更新前と内容が同じなら終了
                break

            pred = new_pred

            # 各クラスタごとに中心を再計算する
            self.cluster_centers_ = np.array([features[pred == i].mean(axis=0)
                                              for i in range(self.n_clusters)])

        return pred

    # 距離を計算用のfunction
    def _euclidean_distance(self, p0, p1):
        return np.sum((p0 - p1) ** 2)

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
