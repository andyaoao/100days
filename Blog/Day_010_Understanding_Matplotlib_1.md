# 10日目Matplotlib検証　Day 10 Understanding Matplotlib

本libraryの勉強目的：KNN分類の結果を2Dで描けること

本日の目標は
1. グラフの種類
2. グラフの構成
3. 異なるデータを同じグラフで出力
4. 異なるマーク記号

## Step 1: グラフの種類
plt.plot 線グラフ  
plt.scatter 散布図  
plt.bar 棒グラフ  
plt.pie 円グラフ  
plt.hist histogram  

## Step 2: グラフの構成
```python
# scatter = 散布図；label = 工人；color = 色
plt.scatter(X_train, y_train, label = "Group1", c="red")

#グラフタイトル
plt.title('Age and EstimatedSalary')

#グラフの軸
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')

# grid lineを描く
plt.grid(True)

#グラフの凡例を表示
plt.legend(loc='upper right')

#設定した内容を出力
plt.show()
```

## Step 3: 異なるデータを同じグラフで出力
```python
# 2回図を設定して、出力する
plt.scatter(X_train, y_train, label = "Group1", c="red")
plt.scatter(X_test, y_test, label = "Group2", c="blue")
plt.show()
```

## Step 4: 異なるマーク記号
```python
markers1 = [".", ",", "o", "v", "^", "<", ">", "1", "2", "3"]
markers2 = ["4", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
markers3 = ["d", "|", "_", "None", None, "", "$x$",
            "$\\alpha$", "$\\beta$", "$\\gamma$"]
for i in x-1:
  plt.scatter(x[i], y1[i], s=300, marker=markers1[i])
  plt.scatter(x[i], y2[i], s=300, marker=markers2[i])
  plt.scatter(x[i], y3[i], s=300, marker=markers3[i])

```

## 補足

### 参考資料
matplotlibの紹介　https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9  
