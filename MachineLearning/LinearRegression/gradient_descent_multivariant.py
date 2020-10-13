import argparse
from datetime import datetime
import sys
import pandas as pd
import numpy as np
from numpy.distutils.fcompiler import str2bool
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


def train(X, Y, model_path, step, threshold, max_loop_num, dynamic_step):

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

        loss_descent_rate = (previous_loss - loss) / previous_loss

        if dynamic_step:
            if 0.00005 < loss_descent_rate <= 0.0005:
                current_step = step / 2
            elif 0.000005 < loss_descent_rate <= 0.00005:
                current_step = step / 10
            elif loss_descent_rate <= 0.000005:
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


def main(input_path, output_path, ignored_columns, preprocess_type, training_data_rate, step_length, threshold_rate, max_loop_num, dynamic_step):
    print("input:", input_path)
    print("output:", output_path)
    print("\n")
    if ignored_columns is not None:
        print("ignored_columns:", ignored_columns)
    print("\n")
    print("preprocess_type:", preprocess_type)
    print("training_data_rate:", training_data_rate)
    print("\n")
    print("threshold_rate:", threshold_rate)
    print("max_loop_num:", max_loop_num)
    print("step_length:", step_length)
    if dynamic_step:
        print("dynamic stepping ...")
    else:
        print("static stepping ...")
    print("\n")
    start_time = datetime.now()

    X, Y = load_data(input_path, True, ignored_columns)

    X = preprocess(X, preprocess_type)

    X_train, y_train, X_test, y_test = split_dataset(X, Y, training_data_rate)

    threshold = gen_threshold(Y, threshold_rate)

    train(X_train, y_train, output_path, step_length, threshold, max_loop_num, dynamic_step)

    Y_pred = predict(output_path, X_test)

    rmse_ration = calculate_rmse_ration(y_test, Y_pred)
    print("rmse ratio (rmse / y_mean) is:", rmse_ration, "\n")

    end_time = datetime.now()

    execution_duration = end_time - start_time

    print("execution duration:", execution_duration, "\n")

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train multivariant LR models with gradient descent algorithm')

    parser.add_argument('--input', metavar='I', type=str, required=True,
                        help='path of input data file')
    parser.add_argument('--output', metavar='O', type=str, required=True,
                        help='path of output model file')

    parser.add_argument('--ignoredColumns', metavar='C', type=str,
                        help='columns ignored from the data file')
    parser.add_argument('--preprocessType', metavar="P", type=str,
                        help="Type of preprocess data, e.g. 'unchange', 'normalize', 'scale'")
    parser.add_argument('--trainingDataRate', metavar="R", type=float,
                        help="rate of data from whole data as training data")
    parser.add_argument('--stepLength', metavar="S", type=float, required=True,
                        help="length of a step, default is 2.0")
    parser.add_argument('--thresholdRate', metavar="T", type=float,
                        help="threshold rate of stopping looping, default is 0.001")
    parser.add_argument('--maxLoopNumber', metavar="M", type=float,
                        help="Max Loop Number, default is 50000")
    parser.add_argument('--dynamicStep', metavar="D", type=str2bool,
                        default=True,
                        help="Dyname stepping or static stepping, default is False")

    args = parser.parse_args()

    ignored_columns = None

    type = "unchange"
    training_data_rate = 0.9
    threshold_rate = 0.001
    max_loop_num = 50000

    dynamic = True

    input_path = args.input
    output_path = args.output

    if not args.ignoredColumns is None:
        ignored_columns = str(args.ignoredColumns).split(",")

    if not args.preprocessType is None:
        type = args.preprocessType

    if not args.trainingDataRate is None:
        training_data_rate = args.trainingDataRate


    step_length = args.stepLength

    if not args.thresholdRate is None:
        threshold_rate = args.thresholdRate

    if not args.maxLoopNumber is None:
        max_loop_num = args.maxLoopNumber

    if not args.dynamicStep is None:
        dynamic = bool(args.dynamicStep)

    main(input_path, output_path, ignored_columns, type, training_data_rate, step_length, threshold_rate, max_loop_num, dynamic)