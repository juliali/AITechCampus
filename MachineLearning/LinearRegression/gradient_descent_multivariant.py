import sys
import pandas as pd
import os
import numpy as np
from utilities import dump_multivariant, load_multivariant, draw_graph, load_housing_data, normalize, split_dataset, calculate_rmse_ration


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
    if len(X) < 1:
        return None

    theta = []
    theta.append(0)

    n = len(X[0])
    if len(X) > 1:
        sum = 0
        for i in range(1, n):
            theta_i = ((Y[1] - Y[0]) / (X[1][i] - X[0][i] + 1))
            #print("theta", i, theta_i[0] )
            theta.append(theta_i[0])
            sum += theta_i[0] * X[0][i]
        theta[0] = (Y[0] - sum)[0]
    else:
        theta = np.zeros(n)

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

    print("initial: theta =", theta)

    loss = sys.maxsize

    loop_num = 0

    current_step = step
    while is_loop(threshold, max_loop_num, loop_num, loss):
        n = len(X[0])

        sum = np.zeros(n)
        for j in range(0, n):
            for i in range(0, m):

                dot_sum_j = 0
                for k in range(0, n):
                    dot_sum_j += theta[k] * X[i][k]

            sum[j] += dot_sum_j * X[i][j]

        for j in range(0, n):
            theta[j] -= current_step * sum[j]/m

        loss = 0

        for i in range(0, m):
            residual = 0
            for j in range(0, n):
                residual += theta[j] * X[i][j]
            residual -= Y[i]

            loss += residual * residual

        loss = loss / m
        if loss < 600:
            current_step = step / 10
        elif loss < 595:
            current_step = step / 100

        loop_num += 1
        print("[loop " + str(loop_num) + "]: loss = " + str(loss))

    dump_multivariant(theta, model_path)


def predict(path, X):
    model = pd.read_csv(path)

    theta = model['theta']
    #print(theta)
    #theta = load_multivariant(path)

    predicted_y = []
    for x in X:
        y = 0
        for j in range(0, len(theta)):
            y += x[j] * theta[j]
        predicted_y.append(y)

    return predicted_y


def main():
    X, Y = load_housing_data('input' + os.sep + 'housing.csv')
    X = normalize(X)

    path = 'output' + os.sep + 'gradient_descent_multivariant.csv'

    step = 10.0
    print("step:", step)

    threshold = gen_threshold(Y, 0.001)
    print("threshold:" , threshold)

    max_loop_num = 1000
    print("max_loop_num:", max_loop_num)

    train(X, Y, path, step, threshold, max_loop_num)

    Y_pred = predict(path, X)

    rmse_ration = calculate_rmse_ration(Y, Y_pred)
    print(rmse_ration)


main()