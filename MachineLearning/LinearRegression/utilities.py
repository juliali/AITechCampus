import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
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


def predict(path, X):
    a, b = load(path)
    predicted_y = []
    for x in X:
        y = a + b * x
        predicted_y.append(y)

    return predicted_y


def normalize(X):
    X_normailized = preprocessing.normalize(X)

    return X_normailized


def load_housing_data(path):
    data = pd.read_csv(path)
    n = len(data['CRIM'])
    X0 = np.ones(n)

    X = [X0,
         data['CRIM'],
         data['ZN'],
         data['INDUS'],
         data['CHAS'],
         data['NOX'],
         data['RM'],
         data['AGE'],
         data['DIS'],
         data['RAD'],
         data['TAX'],
         data['PIRATIO'],
         data['B'],
         data['LSTAT']]

    X = np.mat(X).T
    Y = data['MEDV']
    Y = np.array(Y).reshape((len(Y), 1))

    #print(X)
    return X, Y



def split_dataset(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    train_size=0.9,
                                                    test_size = 0.1,
                                                    random_state=42)
    return  X_train, y_train, X_test, y_test


def calculate_rmse_ration(Y_test, Y_predicted):
    mse = np.mean((Y_test - Y_predicted) ** 2)
    rmse = math.sqrt(mse)

    y_mean = np.mean(Y_test)

    rmse_ration = rmse / y_mean

    return rmse_ration