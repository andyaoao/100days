# 87日目 Logistic Regression Parameter Tuning 1

本日の目標は
1. ロジスティック回帰のパラメータチューニングプロセス

## Step 1: ロジスティック回帰のパラメータチューニングプロセス

### 0. 事前準備
ロジスティック回帰手法を適用する前、欠損値の処理、カテゴリfeatureの処理  
feature scalingが必要。  

### 1. 正規化を確定
Parameter C = 1/λ
λが小さい -> Cが大きい：overfittingの可能性が上がる
λが大きい -> Cが小さい：underfittingの可能性が上がる

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.learning_curve import validation_curve

# cを5種類をテスト対象とする
C_param_range = [0.001,0.01,0.1,1,10,100]

sepal_acc_table = pd.DataFrame(columns = ['C_parameter','Accuracy'])
sepal_acc_table['C_parameter'] = C_param_range

plt.figure(figsize=(10, 10))

j = 0
for i in C_param_range:

    # Apply logistic regression model to training data
    lr = LogisticRegression(penalty = 'l2', C = i,random_state = 0)
    lr.fit(X_train_sepal_std,y_train_sepal)

    # Predict using model
    y_pred_sepal = lr.predict(X_test_sepal_std)

    # Saving accuracy score in table
    sepal_acc_table.iloc[j,1] = accuracy_score(y_test_sepal,y_pred_sepal)
    j += 1

    # Printing decision regions
    plt.subplot(3,2,j)
    plt.subplots_adjust(hspace = 0.4)
    plot_decision_regions(X = X_combined_sepal_standard
                      , y = Y_combined_sepal
                      , classifier = lr
                      , test_idx = range(105,150))
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('C = %s'%i)
```

## 参考資料
https://www.kaggle.com/joparga3/2-tuning-parameters-for-logistic-regression  
https://medium.com/@yehjames/%E8%B3%87%E6%96%99%E5%88%86%E6%9E%90-%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E7%AC%AC3-3%E8%AC%9B-%E7%B7%9A%E6%80%A7%E5%88%86%E9%A1%9E-%E9%82%8F%E8%BC%AF%E6%96%AF%E5%9B%9E%E6%AD%B8-logistic-regression-%E4%BB%8B%E7%B4%B9-a1a5f47017e5  
http://thchou.blogspot.com/2009/03/logistic-regression.html  
https://murphymind.blogspot.com/2017/04/machine-learning-logistic-regression.html  
  
