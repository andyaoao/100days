import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_digits
import numpy.random as r

# day 16 start

# numpyが用意しているデータをロード
digits = load_digits()
plt.gray()
plt.matshow(digits.images[2])
# plt.show()

# 各数値は8X8のpixel(データ)に構成されている
# 64個0-15の数値がある
digits.data[0,:]

# Activationには、0-1の数値を使用するので、feature scaling
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)

# データを分割
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# day 17 start

# output layerの設定
def convert_y_to_vect(y):
    y_vect = np.zeros((len(y), 10))
    for i in range(len(y)):
        y_vect[i, y[i]] = 1
    return y_vect
y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)
# print (y_train[1], y_v_train[1])


# create neural Network
# 3 layer with 64 30 10 neurons
nn_structure = [64, 30, 10]

# sigmoid functionの用意
def f(x):
    return 1 / (1 + np.exp(-x))
def f_deriv(x):
    return f(x) * (1 - f(x))

# ランダムでweightとbiasを初期化
def setup_and_init_weights(nn_structure):
    W = {}
    b = {}
    for l in range(1, len(nn_structure)):
        W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
        b[l] = r.random_sample((nn_structure[l],))
    return W, b

# weightとbiasの平均値合計
def init_tri_values(nn_structure):
    tri_W = {}
    tri_b = {}
    for l in range(1, len(nn_structure)):
        tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
        tri_b[l] = np.zeros((nn_structure[l],))
    return tri_W, tri_b
