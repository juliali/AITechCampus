import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import math
from sklearn.model_selection import train_test_split


def draw_graph(X, y, y_pred):
    # 将测试结果以图标的方式显示出来
    plt.scatter(X, y, color='black')
    plt.plot(X, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


def dump(a, b, path):
    with open(path, "w") as file:
        file.write("a,b" + "\n")
        file.write(str(a) + ',' +  str(b))
    return


def load(path):
    model = pd.read_csv(path)
    a = model['a']
    b = model['b']

    return a, b


def dump_multivariant(theta, path):
    with open(path, "w") as file:
        title = "theta\n"
        content = ""
        for j in range(0, len(theta)):
            content += str(theta[j])
            if j < len(theta) -1:
                content += "\n"
        file.write(title)
        file.write(content)
    return


def load_multivariant(path):
    model = pd.read_csv(path)
    theta = model[[]]

    return theta


def preprocess(X, type = "unchange"):

    X_processed = np.asarray(X)

    if type == "normalize":
        X_processed = preprocessing.normalize(X)
    elif type == "scale":
        X_processed = preprocessing.scale(X)

    return X_processed


def load_data(path, include_x0=True, ignored_columns=None):
    data = pd.read_csv(path)
    cn = len(data.keys())

    X = []

    if include_x0:
        n = len(data[data.keys()[0]])
        X0 = np.ones(n)
        X.append(X0)

    for i in range(0, cn - 1):
        column_name = data.keys()[i]
        if (ignored_columns is None) or (not column_name in ignored_columns):
            X.append(data[column_name])

    X = np.mat(X).T

    Y = data[data.keys()[cn - 1]]
    Y = np.array(Y).reshape((len(Y), 1))

    return X, Y


def split_dataset(X, Y, train_rate = 0.9):
    if train_rate == 1.0:
        X_train = X
        X_test = X
        y_train = Y
        y_test = Y
        return X_train, y_train, X_test, y_test
    else:
        test_rate = 1.0 - float(train_rate)
        X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=train_rate,
                                                    test_size = test_rate,
                                                    random_state=42)
        return X_train, y_train, X_test, y_test


def calculate_rmse_ration(Y_test, Y_predicted):
    mse = np.mean((Y_test - Y_predicted) ** 2)
    rmse = math.sqrt(mse)
    y_mean = np.mean(Y_test)
    rmse_ration = rmse / y_mean

    return rmse_ration