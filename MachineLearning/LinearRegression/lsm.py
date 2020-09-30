import pandas as pd
import os
from utilities import dump, load, draw_graph, predict

def calculate(X, Y):

    len_x = len(X)
    len_y = len(Y)

    if (len_x <= 0 or len_y <= 0 or len_x != len_y):
        print("Input data is invalid.")
        exit(1)

    total_x = 0
    total_y = 0
    total_x_square = 0
    total_x_y = 0

    index = 0

    while index < len_x:
        total_x += X[index]
        total_y += Y[index]
        total_x_square += X[index] * X[index]
        total_x_y += X[index] * Y[index]

        index += 1

    return total_x, total_y, total_x_square, total_x_y



def lsm(X, Y, path):
    total_x, total_y, total_x_square, total_x_y = calculate(X, Y)

    n = len(X)

    b = (total_x_y - (1/n)*total_x *total_y) / (total_x_square - (1/n)*(total_x * total_x))

    a = ((1/n)* total_y - b * (1/n) * total_x)

    dump(a, b, path)

    return


def main():
    data = pd.read_csv('input' + os.sep + 'salary.csv')

    X = data['experience']
    Y = data['salary']

    path = 'output' + os.sep + 'lsm.csv'

    lsm(X, Y, path)

    Y_pred = predict(path, X)

    draw_graph(X, Y, Y_pred)


main()
