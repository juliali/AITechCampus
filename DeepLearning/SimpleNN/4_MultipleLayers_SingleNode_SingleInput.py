import math
import pandas as pd
import numpy as np
import random


class SingleInputSingleNode:
    def __init__(self, alpha):
        self.w = random.random()
        self.b = random.random()
        self.alpha = alpha
        self.delta_loss = 0

        print("init: w: {0}; b: {1}".format(self.w, self.b))
        return

    def __active(self, z):
        output = 1.0 / (1.0 + math.exp(float(-z)))
        return output

    def forward(self, input):
        z = self.w * float(input) + self.b
        a = self.__active(z)
        return a

    def backward(self, input, next_layer_delta_loss, next_layer_w):
        if next_layer_w is None:
            self.delta_loss = next_layer_delta_loss
        else:
            output = self.forward(input)
            self.delta_loss = next_layer_delta_loss * next_layer_w * output * (1 - output)

        self.w -= self.alpha * self.delta_loss * input
        self.b -= self.alpha * self.delta_loss
        return

class MultipleLayer:
    def __init__(self, layers, max_loop_num, loss_threshold):
        self.layers = layers
        self.layer_num = len(layers)
        self.max_loop_num = max_loop_num
        self.loss_threshold = loss_threshold

    def forward(self, x, layer_count):
        output = 0
        input = x

        if layer_count < 0:
            layer_count = self.layer_num

        for i in range(0, layer_count):
            output = self.layers[i].forward(input)
            input = output

        return output

    def backward(self, x, y):

        for layer_index in range(self.layer_num - 1, -1, -1):
            if layer_index > 0:
                input = self.forward(x, layer_index)
            else:
                input = x

            output = self.forward(x, layer_index + 1)
            if layer_index == self.layer_num - 1:
                next_layer_delta_loss = output - y
                next_layer_w = None
            else:
                next_layer_delta_loss = self.layers[layer_index + 1].delta_loss
                next_layer_w = self.layers[layer_index + 1].w

            self.layers[layer_index].backward(input, next_layer_delta_loss, next_layer_w)

        return

    def loss(self, x, y):
        a = self.forward(x, -1)
        loss = -1.0 * (float(y) * math.log(a) + float(1-y)* math.log(1-a))
        return loss

    def epoch(self, list_x, list_y):
        num = len(list_x)
        for i in range(0, num):
            self.backward(list_x[i], list_y[i])

        loss = 0.0
        for i in range(0, num):
            loss += self.loss(list_x[i], list_y[i])

        loss = float(loss) / float(num)
        return loss

    def training(self, list_x, list_y):
        loop = 0
        pre_loss = -1
        while loop < self.max_loop_num:
            loss = self.epoch(list_x, list_y)
            loop += 1

            print("epoch {0} -- loss: {1}; ".format(loop, loss))
            if loss < self.loss_threshold or pre_loss == loss:
                break
            pre_loss = loss
        return

    def predict(self, list_x, list_y):
        num = len(list_x)
        correct = 0
        wrong = 0
        for i in range(0, num):
            a = self.forward(list_x[i], -1)
            real_y = list_y[i]
            if (a > 0.5 and real_y == 1) or (a <= 0.5 and real_y == 0):
                correct += 1
            else:
                wrong += 1

        accuracy = float(correct) / float(num) * 100.0
        print("accuracy: {0}%".format(accuracy))
        return


file_path = "data\\data_banknote_authentication.txt"

data = pd.read_csv(file_path, sep=",", header=None)

data = pd.read_csv(file_path, sep=",", header=None)
list_x = data[0]
list_y = data[4]

alpha = 0.001
nns = []

nns.append(SingleInputSingleNode(alpha))
nns.append(SingleInputSingleNode(alpha))

mnn = MultipleLayer(nns, 500, 0.0001)

mnn.training(list_x, list_y)
mnn.predict(list_x, list_y)


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







