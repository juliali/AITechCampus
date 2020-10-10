import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utilities import load_housing_data, normalize, split_dataset, calculate_rmse_ration

def loadData():
    data = pd.read_csv('input' + os.sep + 'housing.csv')
    X = [data['CRIM'],	data['ZN'],	data['INDUS'],	data['CHAS'],	data['NOX'],	data['RM'],	data['AGE'],	data['DIS'],	data['RAD'],	data['TAX'],	data['PIRATIO'],	data['B'],	data['LSTAT']]
    X = np.mat(X).T
    Y = data['MEDV']
    Y = np.array(Y).reshape((len(Y), 1))
    print(X[0])
    print(Y)


def XY(x, y, order):
    X = []
    for i in range(order + 1):
        X.append(x ** i)
    X = np.mat(X).T
    Y = np.array(y).reshape((len(y), 1))
    return X, Y


def figPlot(x1, y1, x2, y2):
    plt.plot(x1, y1, color='g', linestyle='-', marker='')
    plt.plot(x2, y2, color='m', linestyle='', marker='.')
    plt.show()

def lsm(X, Y):
    XT = X.transpose()  # X的转置
    Theta = np.dot(np.dot(np.linalg.inv(np.dot(XT, X)), XT), Y)  # 套用最小二乘法公式
    return Theta

def main():
    X, Y = load_housing_data('input' + os.sep + 'housing.csv')
    X = normalize(X)

    X_train, y_train, X_test, y_test = split_dataset(X, Y)

    Theta = lsm(X_train, y_train)
    y_predicted = np.dot(X_test, Theta)


    rmse_ration = calculate_rmse_ration(y_test, y_predicted)
    print(rmse_ration)

main()