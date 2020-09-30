import sys
import pandas as pd
import os
from utilities import dump, load, draw_graph, predict


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
    n = 0
    sum = 0
    for y in Y:
        sum += y
        n += 1

    average = sum / n

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

    n = len(X)

    a, b = init(X, Y)

    print("initial: a = " + str(a) + ", b = " + str(b))

    loss = sys.maxsize

    loop_num = 0

    while is_loop(threshold, max_loop_num, loop_num, loss):
        sum_a = 0
        sum_b = 0

        for i in range(0, n):
            sum_a += a + b * X[i] - Y[i]
            sum_b += X[i] * (a + b* X[i] - Y[i])

        print(sum_a/n, sum_b/n)
        a -=  step * sum_a / n
        b -=  step * sum_b / n

        loss = 0

        for i in range(0, n):
            residual = a + b * X[i] - Y[i]
            loss += residual * residual

        loop_num += 1
        print("[loop " + str(loop_num) + "]: a = " + str(a) + ", b = " + str(b) + ", loss = " + str(loss))

    dump(a, b, model_path)


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


main()