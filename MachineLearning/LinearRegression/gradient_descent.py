import sys
import pandas as pd
import os
import numpy as np
from utilities import dump, load, draw_graph, calculate_rmse_ration


def is_loop(threshold, max_loop_num, loop_num, loss):
    looping = True
    if threshold > 0:
        if loss <= threshold:
            looping = False
    if max_loop_num > 0:
        if loop_num >= max_loop_num:
            looping = False

    return looping


def init(X, Y):
    if len(X) > 1:
        b = (Y[1] - Y[0])/(X[1] - X[0])
        a = Y[0] - b * X[0]
    else:
        a = 0
        b = 0

    return a, b


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

    a, b = init(X, Y)

    print("initial: a = " + str(a) + ", b = " + str(b))

    loss = sys.maxsize

    loop_num = 0

    while is_loop(threshold, max_loop_num, loop_num, loss):
        sum_a = 0
        sum_b = 0

        for i in range(0, m):
            sum_a += a + b * X[i] - Y[i]
            sum_b += X[i] * (a + b* X[i] - Y[i])

        print(sum_a/m, sum_b/m)
        a -=  step * sum_a / m
        b -=  step * sum_b / m

        loss = 0

        for i in range(0, m):
            residual = a + b * X[i] - Y[i]
            loss += residual * residual

        loss = loss / m
        loop_num += 1
        print("[loop " + str(loop_num) + "]: a = " + str(a) + ", b = " + str(b) + ", loss = " + str(loss))

    dump(a, b, model_path)


def predict(path, X):
    a, b = load(path)
    predicted_y = []
    for x in X:
        y = a + b * x
        predicted_y.append(y)

    return predicted_y

def main():

    data = pd.read_csv('input' + os.sep + 'salary.csv')

    X = data['experience']
    Y = data['salary']

    path = 'output' + os.sep + 'gradient_descent.csv'

    step = 0.005
    print("step:", step)

    threshold = gen_threshold(Y, 0.001)
    print("threshold:" , threshold)

    max_loop_num = 1000
    print("max_loop_num:", max_loop_num)

    train(X, Y, path, step, threshold, max_loop_num)

    Y_pred = predict(path, X)

    draw_graph(X, Y, Y_pred)

    Y = np.array(Y).reshape((len(Y), 1))
    Y_pred = np.array(Y_pred).reshape((len(Y_pred), 1))

    rmse_ration = calculate_rmse_ration(Y, Y_pred)
    print("rmse ratio:", rmse_ration)
    return


if __name__ == "__main__":
    main()