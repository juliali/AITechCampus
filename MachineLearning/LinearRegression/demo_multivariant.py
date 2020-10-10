import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import linear_model
import pickle
from sklearn import preprocessing
import math
from utilities import load_housing_data, normalize, split_dataset, calculate_rmse_ration


def train(X_train, y_train, model_file_path):

    # 创建线性回归模型
    regr = linear_model.LinearRegression()

    # 用训练集训练模型——看就这么简单，一行搞定训练过程
    regr.fit(X_train, y_train)

    pickle.dump(regr, open(model_file_path, 'wb'))

    return


def predict(X_test, model_file_path):
    regr = pickle.load(open(model_file_path, 'rb'))

    # 用训练得出的模型进行预测
    diabetes_y_pred = regr.predict(X_test)

    return diabetes_y_pred


def main():
    model_file_path = "output" + os.sep + "linear_regression_model_mv.sav"

    X, Y = load_housing_data('input' + os.sep + 'housing.csv')
    X = normalize(X)

    X_train, y_train, X_test, y_test = split_dataset(X, Y)

    train(X_train, y_train, model_file_path)
    y_predicted = predict(X_test, model_file_path)

    rmse_ration = calculate_rmse_ration(y_test, y_predicted)

    print(rmse_ration)

main()
