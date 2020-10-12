import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from utilities import load_data, preprocess, split_dataset, calculate_rmse_ration, dump_multivariant, load_multivariant


def lsm(X, Y, path):
    XT = X.transpose()  # X的转置
    Theta = np.dot(np.dot(np.linalg.inv(np.dot(XT, X)), XT), Y)  # 套用最小二乘法公式

    theta = Theta.reshape(1, len(Theta))[0]
    dump_multivariant(theta, path)
    return Theta


def predict(path, X):
    model = pd.read_csv(path)
    theta = model['theta']
    Theta = np.array(theta).reshape((len(theta), 1))

    Y_pred = np.dot(X, Theta)
    Y_pred = np.array(Y_pred).reshape((len(Y_pred), 1))
    return Y_pred


def main():
    X, Y = load_data('input' + os.sep + 'housing.csv')

    X = preprocess(X)

    X_train, y_train, X_test, y_test = split_dataset(X, Y)

    path = 'output' + os.sep + 'lsm_multivariant.csv'

    lsm(X_train, y_train, path)
    y_predicted = predict(path, X_test)

    rmse_ration = calculate_rmse_ration(y_test, y_predicted)
    print("rmse ratio is:", rmse_ration)


main()