import sys
import pandas as pd
import os
import numpy as np
from utilities import dump_multivariant, load_multivariant, draw_graph, load_data, preprocess, split_dataset, calculate_rmse_ration


def is_loop(threshold, max_loop_num, loop_num, previous_loss, loss):
    looping = True
    if previous_loss < loss:
        looping = False
    if threshold > 0:
        if loss <= threshold:
            looping = False
    if max_loop_num > 0:
        if loop_num >= max_loop_num:
            looping = False

    return looping


def init(X, Y):
    if len(X) < 1:
        return None

    n = len(X[0].T)
    #print(n, X[0])
    theta = np.zeros(n)

    if len(X) > 1:
        sum = 0
        for i in range(1, n):
            divide = X[1][i] - X[0][i]
            if divide == 0:
                divide = 0.01

            theta_i = (Y[1] - Y[0]) / divide
            theta[i] = theta_i
            sum += theta_i[0] * X[0][i]
        theta[0] = (Y[0] - sum)[0]

    return theta


def gen_threshold(Y, ratio):
    m = 0
    sum = 0
    for y in Y:
        sum += y
        m += 1

    average = sum / m

    residual_duration = (average * ratio)
    threshold = residual_duration * residual_duration

    return threshold


def train(X, Y, model_path, step, threshold, max_loop_num):

    if len(X) <= 0 or len(Y) <=0 or len(X) != len(Y):
        print("Input data invalid!")
        exit(1)

    if step < 0:
        print("Step must be larger than 0.")
        exit(1)

    if threshold <= 0 and max_loop_num <= 0:
        print("threshold and max_loop_num must be set as least one.")
        exit(1)

    m = len(X)

    theta = init(X, Y)

    print("\ninitial: theta =", theta, "\n")

    loss = sys.maxsize

    loop_num = 0
    previous_loss = loss

    current_step = step

    while is_loop(threshold, max_loop_num, loop_num, previous_loss, loss):
        previous_loss = loss
        n = len(X[0])
        sum = np.zeros(n)
        for j in range(0, n):
            sum[j] = 0
            for i in range(0, m):
                sum[j] += (np.dot(X[i], theta) - Y[i]) * X[i][j]

        for j in range(0, n):
            theta[j] -= current_step * sum[j]/m

        loss = 0
        for i in range(0, m):
            residual = np.dot(X[i], theta) - Y[i][0]
            loss += residual * residual

        loss = loss / m

        loss_decsend_rate = (previous_loss - loss) / previous_loss

        if 0.00005 < loss_decsend_rate <= 0.0005:
            current_step = step / 2
        elif 0.000005 < loss_decsend_rate <= 0.00005:
            current_step = step / 10
        elif loss_decsend_rate <= 0.000005:
            current_step = step / 100

        loop_num += 1
        print("[loop " + str(loop_num) + "]: loss = " + str(loss))

    dump_multivariant(theta, model_path)
    print("\n")
    return


def predict(path, X):
    model = pd.read_csv(path)

    theta = model['theta']

    Y_pred = []
    for x in X:
        y = np.dot(x, theta)
        Y_pred.append(y)

    Y_pred = np.array(Y_pred).reshape((len(Y_pred), 1))

    return Y_pred


def main():
    ignored_columns = ['ZN','CHAS','NOX','RM','DIS','RAD','TAX','PIRATIO','B','LSTAT']
    X, Y = load_data('input' + os.sep + 'housing.csv', True, ignored_columns)

    X = preprocess(X, "normalize")

    X_train, y_train, X_test, y_test = split_dataset(X, Y)

    path = 'output' + os.sep + 'gradient_descent_multivariant.csv'

    step = 2.0 #0.00005
    print("step:", step)

    threshold = gen_threshold(Y, 0.001)
    print("threshold:", threshold)

    max_loop_num = 30000
    print("max_loop_num:", max_loop_num)

    train(X_train, y_train, path, step, threshold, max_loop_num)

    Y_pred = predict(path, X_test)

    rmse_ration = calculate_rmse_ration(y_test, Y_pred)
    print("rmse ration is:", rmse_ration)


main()