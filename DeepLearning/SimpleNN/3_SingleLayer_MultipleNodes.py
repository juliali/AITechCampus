import math
import pandas as pd
import numpy as np


class SingleLayer:
    def __init__(self, feature_num, node_num, alpha, max_loop_num, loss_threshold):
        self.w = np.random.uniform(size=(node_num, feature_num + 1))
        self.feature_num = feature_num
        self.node_num = node_num
        self.alpha = alpha
        self.max_loop_num = max_loop_num
        self.loss_threshold = loss_threshold

        print("init: w: ", self.w)
        return

    def __active(self, Z):
        outputs = []
        for i in range(0, self.node_num):
            output = 1.0 / (1.0 + math.exp(float(-Z[i])))
            outputs.append(output)
        return outputs

    def __forward(self, X):
        Z = []
        for i in range(0, self.node_num):
            z = self.w[i][0]
            for j in range(0, self.feature_num):
                z += self.w[i][j+1] * float(X[j])
            Z.append(z)
        A = self.__active(Z)
        return A

    def __backward(self, X, Y):
        for i in range(0, self.node_num):
            for j in range(0, self.feature_num):
                self.w[i][j + 1] -= self.alpha * (self.__forward(X)[i] - float(Y[i])) * float(X[j])
            self.w[i][0] -= self.alpha * (self.__forward(X)[i] - Y[i])
        return

    def __loss(self, X, Y):
        Loss = []
        A = self.__forward(X)
        for i in range(0, self.node_num):
            loss = -1.0 * (float(Y[i]) * math.log(A[i]) + float(1-Y[i])* math.log(1-A[i]))
            Loss.append(loss)
        return Loss

    def __epoch(self, list_X, list_Y):
        num = len(list_X)
        for i in range(0, num):
            self.__backward(list_X[i], list_Y[i])

        Loss = [0.0, 0.0]
        for i in range(0, num):
            tmp_Loss = self.__loss(list_X[i], list_Y[i])

            for j in range(0, self.node_num):
                tmp_Loss[j] = float(tmp_Loss[j]) / float(num)
                Loss[j] += tmp_Loss[j]

        return Loss

    def training(self, list_X, list_Y):
        loop = 0
        pre_Loss = [-1, -1]
        while loop < self.max_loop_num:
            Loss = self.__epoch(list_X, list_Y)
            loop += 1

            print("epoch {0} -- loss: {1}".format(loop, Loss))
            if Loss < self.loss_threshold or pre_Loss == Loss:
                break
            pre_Loss = Loss
        return

    def predict(self, list_X, list_Y):
        num = len(list_X)
        correct = 0
        wrong = 0
        for i in range(0, num):
            A = self.__forward(list_X[i])
            real_Y = list_Y[i]

            isEqual = True
            for j in range(0, self.node_num):
                if A[j] >= 0.5:
                    A[j] = 1.0
                else:
                    A[j] = 0.0

                if int(A[j]) != real_Y[j]:
                    isEqual = False
                    break
            if isEqual:
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

node_num = 2

list_Y = []
for i in range(0, len(list_y)):
    Y = []
    Y.append(list_y[i])
    for j in range(1, node_num):
        Y.append(0)
    list_Y.append(Y)


print(list_Y)
nn = SingleLayer(4, 2, 0.001, 500, [0.005, 0.005])
nn.training(list_X,list_Y)
nn.predict(list_X, list_Y)







