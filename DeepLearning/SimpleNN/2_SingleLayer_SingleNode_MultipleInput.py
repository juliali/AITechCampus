import math
import pandas as pd
import numpy as np


class MultipleInputSingleNode:
    def __init__(self, feature_num, alpha, max_loop_num, loss_threshold):
        self.w = np.random.uniform(size=(feature_num + 1,))
        self.feature_num = feature_num
        self.alpha = alpha
        self.max_loop_num = max_loop_num
        self.loss_threshold = loss_threshold

        print("init: w: ", self.w)
        return

    def __active(self, z):
        output = 1.0 / (1.0 + math.exp(float(-z)))
        return output

    def __forward(self, X):
        z = self.w[0]
        for i in range(0, self.feature_num):
            z += self.w[i+1] * float(X[i])
        a = self.__active(z)
        return a

    def __backward(self, X, y):
        for i in range(0, self.feature_num):
            self.w[i + 1] -= self.alpha * (self.__forward(X) - float(y)) * float(X[i])
        self.w[0] -= self.alpha * (self.__forward(X) - y)

        return

    def __loss(self, X, y):
        a = self.__forward(X)
        loss = -1.0 * (float(y) * math.log(a) + float(1-y)* math.log(1-a))
        return loss

    def __epoch(self, list_X, list_y):
        num = len(list_X)
        for i in range(0, num):
            self.__backward(list_X[i], list_y[i])

        loss = 0.0
        for i in range(0, num):
            loss += self.__loss(list_X[i], list_y[i])

        loss = float(loss) / float(num)
        return loss

    def training(self, list_X, list_y):
        loop = 0
        pre_loss = -1
        while loop < self.max_loop_num:
            loss = self.__epoch(list_X, list_y)
            loop += 1

            print("epoch {0} -- loss: {1}".format(loop, loss))
            if loss < self.loss_threshold or pre_loss == loss:
                break
            pre_loss = loss
        return

    def predict(self, list_X, list_y):
        num = len(list_X)
        correct = 0
        wrong = 0
        for i in range(0, num):
            a = self.__forward(list_X[i])
            real_y = list_y[i]
            if (a >= 0.5 and real_y == 1) or (a < 0.5 and real_y == 0):
                correct += 1
            else:
                wrong += 1

        accuracy = float(correct) / float(num) * 100.0
        print("accuracy: {0}%".format(accuracy))
        return


file_path = "data\\data_banknote_authentication.txt"

data = pd.read_csv(file_path, sep=",", header=None)

list_X = np.stack([data[0].values, data[1].values, data[2].values, data[3].values], axis=1)
list_X = list_X.reshape(-1, 4)

list_y = data[4]

nn = MultipleInputSingleNode(4, 0.001, 500, 0.005)
nn.training(list_X,list_y)
nn.predict(list_X, list_y)







