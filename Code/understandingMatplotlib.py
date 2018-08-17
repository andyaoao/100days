import numpy as np
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

plt.show()
