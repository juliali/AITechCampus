import math
import pandas as pd
import numpy as np
import random


class MultipleInputSingleNode:
    def __init__(self, feature_num, alpha):
        self.w = np.random.uniform(size=(feature_num + 1,))
        self.feature_num = feature_num
        self.alpha = alpha
        self.delta_loss = np.random.uniform(size=(feature_num + 1,))#np.atleast_1d(0)

        print("init: w: {0};".format(self.w))
        return

    def __active(self, z):
        output = 1.0 / (1.0 + math.exp(float(-z)))
        return output

    def forward(self, Input):
        z = self.w[0]
        for i in range(0, self.feature_num):
            z += self.w[i + 1] * float(Input[i])
        a = self.__active(z)
        return np.atleast_1d(a)

    def backward(self, Input, next_layer_delta_loss, next_layer_w):
        if next_layer_w is None:
            self.delta_loss = next_layer_delta_loss
        else:
            output = self.forward(Input)
            for i in range(1, len(next_layer_delta_loss)):
                for j in range(0, self.feature_num + 1):
                    self.delta_loss[j] = next_layer_delta_loss[i] * next_layer_w[i] * output[0] * (1 - output[0])  ### TODO

        for i in range(1, self.feature_num + 1):
            self.w[i] -= self.alpha * self.delta_loss[i] * Input[i-1]
        self.w[0] -= self.alpha * self.delta_loss[0]

        return


class MultipleLayer:
    def __init__(self, layers, max_loop_num, loss_threshold):
        self.layers = layers
        self.layer_num = len(layers)
        self.max_loop_num = max_loop_num
        self.loss_threshold = loss_threshold

    def forward(self, X, layer_count):
        output = 0
        input = X

        if layer_count < 0:
            layer_count = self.layer_num

        for i in range(0, layer_count):
            output = self.layers[i].forward(input)
            input = output

        return output

    def backward(self, X, y):
        for layer_index in range(self.layer_num - 1, -1, -1):
            if layer_index > 0:
                input = self.forward(X, layer_index)
            else:
                input = X

            output = self.forward(X, layer_index + 1)

            if layer_index == self.layer_num - 1:
                loss_dimensions = 2
                next_layer_delta_loss = np.random.uniform(size=(loss_dimensions,))
                for i in range(0, loss_dimensions):
                    next_layer_delta_loss[i] = output[0] - y
                next_layer_w = None
            else:
                next_layer_delta_loss = self.layers[layer_index + 1].delta_loss
                next_layer_w = self.layers[layer_index + 1].w

            self.layers[layer_index].backward(input, next_layer_delta_loss, next_layer_w)

        return

    def loss(self, X, y):
        a = self.forward(X, -1)
        loss = -1.0 * (float(y) * math.log(a[0]) + float(1-y)* math.log(1-a[0]))
        return np.atleast_1d(loss)

    def epoch(self, list_X, list_y):
        num = len(list_X)
        for i in range(0, num):
            self.backward(list_X[i], list_y[i])

        total_loss = 0.0
        for i in range(0, num):
            total_loss += self.loss(list_X[i], list_y[i])[0]

        total_loss = float(total_loss) / float(num)
        return np.atleast_1d(total_loss)

    def training(self, list_X, list_y):
        loop = 0
        pre_loss = np.atleast_1d(-1.0)
        while loop < self.max_loop_num:
            loss = self.epoch(list_X, list_y)
            loop += 1

            print("epoch {0} -- loss: {1}; ".format(loop, loss))
            if loss[0] < self.loss_threshold or pre_loss == loss:
                break
            pre_loss = loss
        return

    def predict(self, list_X, list_y):
        num = len(list_X)
        correct = 0
        wrong = 0
        for i in range(0, num):
            a = self.forward(list_X[i], -1)
            real_y = list_y[i]
            if (a[0] > 0.5 and real_y == 1) or (a[0] <= 0.5 and real_y == 0):
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

alpha = 0.001
nns = []

nns.append(MultipleInputSingleNode(4, alpha))
nns.append(MultipleInputSingleNode(1, alpha))

mnn = MultipleLayer(nns, 500, 0.0001)

mnn.training(list_X, list_y)
mnn.predict(list_X, list_y)


#list_X = np.stack([data[0].values, data[1].values, data[2].values, data[3].values], axis=1)
#list_X = list_X.reshape(-1, 4)

#list_y = data[4]

#node_num = 2

#list_Y = []
#for i in range(0, len(list_y)):
#    Y = []
#    Y.append(list_y[i])
#    for j in range(1, node_num):
#        Y.append(0)
#    list_Y.append(Y)


#print(list_Y)
#nn = SingleLayer(4, 2, 0.001, 500, [0.005, 0.005])
#nn.training(list_X,list_Y)
#nn.predict(list_X, list_Y)







