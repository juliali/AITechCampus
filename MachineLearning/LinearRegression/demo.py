import numpy as np
from sklearn import linear_model
import os
import pickle
import pandas as pd
import math
from utilities import draw_graph, calculate_rmse_ration

def train(X_train, y_train, model_file_path):

    # 创建线性回归模型
    regr = linear_model.LinearRegression()

    # 用训练集训练模型——看就这么简单，一行搞定训练过程
    regr.fit(X_train, y_train)

    pickle.dump(regr, open(model_file_path, 'wb'))


def predict(X_test, model_file_path):
    regr = pickle.load(open(model_file_path, 'rb'))

    # 用训练得出的模型进行预测
    diabetes_y_pred = regr.predict(X_test)

    return diabetes_y_pred


def main():
    data = pd.read_csv('input' + os.sep + 'salary.csv')

    experiences = np.array(data['experience'])
    salaries = np.array(data['salary'])

    # 将特征数据集分为训练集和测试集，除了最后 4 个作为测试用例，其他都用于训练
    X_train = experiences[:7]
    X_train = X_train.reshape(-1,1)
    X_test = experiences[7:]
    X_test = X_test.reshape(-1,1)

    # 把目标数据（特征对应的真实值）也分为训练集和测试集
    y_train = salaries[:7]
    y_test = salaries[7:]

    model_file_path = "output" + os.sep + "linear_regression_model.sav"

    train(X_train, y_train, model_file_path)
    y_predicted = predict(X_test, model_file_path)

    rmse_ration = calculate_rmse_ration(y_test, y_predicted)
    print(rmse_ration)

    draw_graph(X_test, y_test, y_predicted)

main()