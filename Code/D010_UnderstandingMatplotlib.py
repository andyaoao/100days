import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#day 10 start

#5列のデータセットを読み込む
dataset = pd.read_csv('./Datasets/Social_Network_Ads.csv')
# AgeをXに入れる
X = dataset['Age']
# EstimatedSalaryをYに入れる
y = dataset['EstimatedSalary']

# データを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# scatter = 散布図；label = 工人；color = 色
plt.scatter(X_train, y_train, label = "Group1", c="red")
plt.scatter(X_test, y_test, label = "Group2", c="blue")

# グラフタイトル
plt.title('Age and EstimatedSalary')

# グラフの軸
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')

# grid lineを描く
plt.grid(True)

# グラフの凡例を表示
plt.legend(loc='upper right')

#day 11 start

fig = plt.figure()  # Figure
fig.patch.set_facecolor('blue')  # 図全体の背景色
fig.patch.set_alpha(0.5)  # 図全体の背景透明度

ax = fig.add_subplot(221)  # Axes
ax.patch.set_facecolor('green')  # subplotの背景色
ax.patch.set_alpha(0.3)  # subplotの背景透明度

ax = fig.add_subplot(222)  # Axes
ax.patch.set_facecolor('red')  # subplotの背景色
ax.patch.set_alpha(0.3)  # subplotの背景透明度

ax = fig.add_subplot(223)  # Axes
ax.patch.set_facecolor('yellow')  # subplotの背景色
ax.patch.set_alpha(0.9)  # subplotの背景透明度

plt.show()
