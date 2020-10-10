import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utilities import load_housing_data, normalize, split_dataset, calculate_rmse_ration

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