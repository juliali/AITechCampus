import random
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit


class FCLayer():
    def __init__(self, is_last_layer, node_num, input_feature_num, step_size):
        self.stepSize = step_size
        self.isLastLayer = is_last_layer
        self.nodeNum = node_num

        self.W = np.array([random.random() for v in range(input_feature_num * node_num)]).reshape(node_num, input_feature_num)
        self.B = np.array([random.random() for v in range(node_num * 1)])

        return

    def forward(self, current_input_data):
        M = np.dot(self.W, current_input_data)
        Z = np.add(M, self.B)
        A = expit(Z)
        return A

    def backward(self, current_input_data, dZNext, WNext):
        A = self.forward(current_input_data)

        if self.isLastLayer:
            Y = dZNext
            dZ = np.subtract(A, Y)
        else:
            I = np.ones(len(A))
            IA = np.subtract(I, A)

            dZ = np.dot(dZNext, WNext)
            dZ = np.multiply(dZ, A)
            dZ = np.multiply(dZ, IA)

        dW = np.dot(dZ.reshape(-1, 1), current_input_data.reshape(1, -1))
        dB = dZ

        self.W = np.subtract(self.W, np.multiply(self.stepSize, dW))
        self.B = np.subtract(self.B, np.multiply(self.stepSize, dB))

        return dZ, self.W


class FCNN():
    def __init__(self):
        self.layerArray = []
        return

    def add_layer(self, layer):
        self.layerArray.append(layer)
        return

    def forward(self, input_data):
        current_input = input_data
        for aLayer in self.layerArray:
            current_output = aLayer.forward(current_input)
            current_input = current_output
        return current_output

    def backward(self, input_data, label_data):
        layer_num = len(self.layerArray)
        dZNext = label_data
        WNext = None
        layer_index = layer_num - 1

        while layer_index >= 0:
            current_input = input_data
            if layer_index > 0:
                for i in range(0, layer_index):
                    current_input = self.layerArray[i].forward(current_input)
            current_layer = self.layerArray[layer_index]
            dZNext, WNext = current_layer.backward(current_input, dZNext, WNext)
            layer_index -= 1

        return

    def count_loss(self, outputs, labels):
        crow = outputs.shape[0]
        ccol = outputs.shape[1]

        total = 0.0
        number = 0

        for i in range(0, crow):
            for j in range(0, ccol):
                a = outputs[i][j]
                b = labels[i][j]
                if b == 0:
                    b = 0.00001
                if b == 1:
                    b = 0.99999
                current_loss = -(a * math.log(b) - (1.0 - a) * math.log(1.0 - b))
                if current_loss < 0.0:
                    current_loss = 0 - current_loss

                total += current_loss
                number += 1

        loss = total/ float(number)

        return loss

    def is_equal_to_label(self, output, label):
        if len(output) != len(label):
            raise Exception("output has different features with label -- output:", output, "label:", label)
        dim = len(output)

        for i in range(0, dim):
            if output[i] >= 0.5: # output[i] is set to 0
                if label[i] != 1:
                    return False

        return True


def draw_graph(loss_dict):
    xpoints = np.array(list(loss_dict.keys()))
    ypoints = np.array(list(loss_dict.values()))

    plt.plot(xpoints, ypoints)
    plt.show()
    return


def trainFCNN(input_datas, label_datas, nn, max_epoch_num, threshold):
    loss = 100.0
    loop_num = 0

    loss_dict = {}
    while loss > threshold and loop_num < max_epoch_num:
        outputs = []
        index = 0

        while index < len(input_datas):
            input_data = input_datas[index]
            output = nn.forward(input_data)
            outputs.append(output)
            nn.backward(input_data, label_datas[index])
            index += 1
        outputs = np.array(outputs)
        loss = nn.count_loss(outputs, label_datas)
        loop_num += 1
        loss_dict[loop_num] = loss

    return loss_dict


def predictFCNN(nn, input_datas, label_datas):
    outputs = []
    case_num = len(input_datas)
    index = 0

    correct_num = 0
    while index < case_num:
        output = nn.forward(input_datas[index])
        outputs.append(output)

        if nn.is_equal_to_label(output, label_datas[index]):
            correct_num += 1

        index += 1

    outputs = np.array(outputs)

    loss = nn.count_loss(label_datas, outputs)
    accuracy = (correct_num * 1.0) / (case_num * 1.0) * 100.0

    return loss, accuracy


def generate_labels(raw_labels, dim_num):
    if dim_num < 1:
        raise Exception("Dimension of label cannot be less than 1.")
    else:
        labels = []
        for label in raw_labels:
            new_label = np.array([0] * dim_num)
            if label != 0:
                new_label[0] = label

            labels.append(new_label)

        labels = np.array(labels)
        labels = labels.reshape(-1, dim_num)

        return labels


def test_case(file_path):
    data = pd.read_csv(file_path, sep=",", header=None)

    feature_num = 4
    list_X = np.stack([data[0].values, data[1].values, data[2].values, data[3].values], axis=1)
    list_X = list_X.reshape(-1, feature_num)

    label_dim = 2
    list_y = generate_labels(data[4], label_dim)

    nn = FCNN()

    layer_one_node_num = 3
    layer_one = FCLayer(False, layer_one_node_num, feature_num,  0.05)
    nn.add_layer(layer_one)

    layer_two_node_num = 4
    layer_two = FCLayer(False, layer_two_node_num, layer_one_node_num, 0.05)
    nn.add_layer(layer_two)

    layer_three_node_num = label_dim
    layer_three = FCLayer(True, layer_three_node_num, layer_two_node_num, 0.05)
    nn.add_layer(layer_three)

    loss_dict = trainFCNN(list_X, list_y, nn, 10, 0.01)
    draw_graph(loss_dict)
    loss, accuracy = predictFCNN(nn, list_X, list_y)
    print("Total Loss:", loss, "Accuracy:", str(accuracy) + "%")


file_path = "data\\data_banknote_authentication.txt"
test_case(file_path)