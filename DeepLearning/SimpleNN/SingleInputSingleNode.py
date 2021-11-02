import math
import random
import pandas as pd

class SingleInputSingleNode:
    def __init__(self, alpha, max_loop_num, loss_threshold):
        self.w = random.random()
        self.b = random.random()
        self.alpha = alpha
        self.max_loop_num = max_loop_num
        self.loss_threshold = loss_threshold

        print("init: w: {0}; b: {1}".format(self.w, self.b))
        return

    def __active(self, z):
        output = 1.0 / (1.0 + math.exp(float(-z)))
        return output

    def __forward(self, x):
        z = self.w * float(x) + self.b
        a = self.__active(z)
        return a

    def __backward(self, x, y):
        self.w -= self.alpha * (self.__forward(x) - y) * x
        self.b -= self.alpha * (self.__forward(x) - y)

        return

    def __loss(self, x, y):
        a = self.__forward(x)
        loss = -1.0 * (float(y) * math.log(a) + float(1-y)* math.log(1-a))
        return loss

    def __epoch(self, list_x, list_y):
        num = len(list_x)
        for i in range(0, num):
            self.__backward(list_x[i], list_y[i])

        loss = 0.0
        for i in range(0, num):
            loss += self.__loss(list_x[i], list_y[i])

        loss = float(loss) / float(num)
        return loss

    def training(self, list_x, list_y):
        loop = 0
        pre_loss = -1
        while loop < self.max_loop_num:
            loss = self.__epoch(list_x, list_y)
            loop += 1

            print("epoch {0} -- loss: {1}; w: {2}; b: {3}".format(loop, loss, nn.w, nn.b))
            if loss < self.loss_threshold or pre_loss == loss:
                break
            pre_loss = loss
        return

    def predict(self, list_x, list_y):
        num = len(list_x)
        correct = 0
        wrong = 0
        for i in range(0, num):
            a = self.__forward(list_x[i])
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
list_x = data[0]
list_y = data[4]

nn = SingleInputSingleNode(0.001, 500, 0.01)
nn.training(list_x,list_y)
nn.predict(list_x, list_y)







