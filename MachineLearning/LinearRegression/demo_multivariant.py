import os
from sklearn import linear_model
import pickle
import numpy as np

from utilities import load_data, preprocess, split_dataset, calculate_rmse_ration


def train(X_train, y_train, model_file_path):

    # 创建线性回归模型
    regr = linear_model.LinearRegression()

    # 用训练集训练模型——看就这么简单，一行搞定训练过程
    regr.fit(X_train, y_train)

    pickle.dump(regr, open(model_file_path, 'wb'))

    return


def predict(X_test, model_file_path):
    regr = pickle.load(open(model_file_path, 'rb'))

    # 用训练得出的模型进行预测
    Y_pred = regr.predict(X_test)
    Y_pred = np.array(Y_pred).reshape((len(Y_pred), 1))
    return Y_pred


def main():
    model_file_path = "output" + os.sep + "linear_regression_model_mv.sav"

    ignored_columns = ['ZN', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD', 'TAX', 'PIRATIO', 'B', 'LSTAT']
    X, Y = load_data('input' + os.sep + 'housing.csv', False, ignored_columns)

    X = preprocess(X, "normalize")

    X_train, y_train, X_test, y_test = split_dataset(X, Y)

    train(X_train, y_train, model_file_path)
    y_predicted = predict(X_test, model_file_path)

    rmse_ration = calculate_rmse_ration(y_test, y_predicted)

    print("rmse ratio:", rmse_ration)


if __name__ == "__main__":
    main()
