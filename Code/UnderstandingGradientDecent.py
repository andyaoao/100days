import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#day 7 start

def compute_cost(features, values, weight):
    """
    Compute the cost of a list of parameters, weight, given a list of features
    (input data points) and values (output data points).
    """
    # テストのデータ数
    m = len(values)
    # MSEの合計を計算
    sum_of_square_errors = np.square(np.dot(features, weight) - values).sum()
    # コスト（MSEの合計 / テストデータ数の2倍
    cost = sum_of_square_errors / (2*m)

    return cost

def gradient_descent(features, values, weight, alpha, num_iterations):
    """
    Perform gradient descent given a data set with an arbitrary number of features.
    """

    # コストの推移を記録する
    cost_history = []

    # training iterationsは実行の回数
    for i in range(0, num_iterations):
        # 毎回計算したコストを陣列に保存
        cost_history.append(compute_cost(features, values, weight))
        # 関数での計算結果
        hypothesis = np.dot(features, weight)
        # 関数の計算結果とテスト数値の差はロスと扱う
        loss = hypothesis - values
        # gradientを計算
        gradient = np.dot(features.transpose(), loss) / len(values)
        # gradientにより、weightを修正
        weight = weight - alpha*gradient
    return weight, cost_history


# 2列のデータセットを読み込む
dataset = pd.read_csv('./Datasets/studentscores.csv')
X = dataset[dataset.columns[0]]
Y = dataset[dataset.columns[1]]

# 75-25でtraining setとtest setを分割
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 0.25, random_state = 0)
gradient_result = gradient_descent(X_train, Y_train, 0.01, 0.01, 5)

# 図でコストが学習の回数により下がることが確認できる
plt.plot(gradient_result[1])
plt.show()
