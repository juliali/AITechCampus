import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

experiences = np.array([0,1,2,3,4,5,6,7,8,9,10])
salaries = np.array([103100, 104900, 106800, 108700, 110400, 112300, 114200, 116100, 117800, 119700, 121600])

# 将特征数据集分为训练集和测试集，除了最后 4 个作为测试用例，其他都用于训练
X_train = experiences[:7]
X_train = X_train.reshape(-1,1)
X_test = experiences[7:]
X_test = X_test.reshape(-1,1)

# 把目标数据（特征对应的真实值）也分为训练集和测试集
y_train = salaries[:7]
y_test = salaries[7:]

# 创建线性回归模型
regr = linear_model.LinearRegression()

# 用训练集训练模型——看就这么简单，一行搞定训练过程
regr.fit(X_train, y_train)

# 用训练得出的模型进行预测
diabetes_y_pred = regr.predict(X_test)

# 将测试结果以图标的方式显示出来
plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()