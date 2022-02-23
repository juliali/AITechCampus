import random
import numpy as np
import math
import pandas as pd
from scipy.special import expit


class FCLayer():
    def __init__(self, isLastLayer, nodeNum, inputVectorSize, stepSize):
        self.stepSize = stepSize
        self.isLastLayer = isLastLayer
        self.nodeNum = nodeNum

        self.W = np.array([random.random() for v in range(inputVectorSize * nodeNum)]).reshape(nodeNum, inputVectorSize)
        self.B = np.array([random.random() for v in range(nodeNum * 1)])

        return

    def forward(self, currentInputData):
        M = np.dot(self.W, currentInputData)
        Z = np.add(M, self.B)
        A = expit(Z)
        return A

    def backward(self, currentInputData, dZNext, WNext):
        A = self.forward(currentInputData)

        if self.isLastLayer:
            Y = dZNext
            dZ = np.subtract(A, Y)
        else:
            I = np.ones(len(A))
            IA = np.subtract(I, A)

            dZ = np.dot(dZNext, WNext)
            dZ = np.multiply(dZ, A)
            dZ = np.multiply(dZ, IA)

        dW = np.dot(dZ.reshape(-1,1),currentInputData.reshape(1,-1))
        dB = dZ

        self.W = np.subtract(self.W, np.multiply(self.stepSize, dW))
        self.B = np.subtract(self.B, np.multiply(self.stepSize, dB))

        return dZ, self.W


class FCNN():
    def __init__(self):
        self.layerArray = []
        return

    def addLayer(self, aLayer):
        self.layerArray.append(aLayer)
        return


    def forward(self, inputData):
        currentInput = inputData
        for aLayer in self.layerArray:
            currentOutput = aLayer.forward(currentInput)
            currentInput = currentOutput
        return currentOutput


    def backward(self, inputData, labelData):
        layerNum = len(self.layerArray)
        dZNext = labelData
        WNext = None
        layerIndex = layerNum - 1

        while layerIndex >= 0:
            currentInput = inputData
            if layerIndex > 0:
                for i in range(0, layerIndex):
                    currentInput = self.layerArray[i].forward(currentInput)
            aLayer = self.layerArray[layerIndex]
            dZNext, WNext = aLayer.backward(currentInput, dZNext, WNext)
            layerIndex -= 1

        return

    def countLoss(self, outputs, labels):
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
                currentLoss = -(a * math.log(b) - (1.0 - a) * math.log(1.0 - b))
                if currentLoss < 0.0:
                    currentLoss = 0 - currentLoss

                total += currentLoss
                number += 1

        loss = total/ float(number)

        return loss


def trainFCNN(inputDatas, labelDatas, nn, maxEpochNum, threshold):
    loss = 100.0
    loopNum = 0

    while loss > threshold and loopNum < maxEpochNum:
        outputs = []
        index = 0

        while index < len(inputDatas):
            inputData = inputDatas[index]
            output = nn.forward(inputData)
            outputs.append(output)
            nn.backward(inputData, labelDatas[index])
            index += 1
        outputs = np.array(outputs)
        loss = nn.countLoss(outputs, labelDatas)
        loopNum += 1
        print("loop:", loopNum, "loss:", loss)

    return #nn


def predictFCNN(nn, inputDatas, labelDatas):
    outputs = []
    caseNum = len(inputDatas)
    index = 0

    correctNum = 0
    while index < caseNum:
        output = nn.forward(inputDatas[index])
        outputs.append(output)

        if isEqualLabel(output, labelDatas[index]):
            correctNum += 1

        index += 1

    outputs = np.array(outputs)
    loss = nn.countLoss(labelDatas, outputs)

    accuracy = (correctNum * 1.0) / (caseNum * 1.0) * 100.0
    print("Total Loss:", loss, "Accurancy:", str(accuracy) + "%")
    return

def isEqualLabel(output, label):
    if len(output) != len(label):
        raise Exception("output has different features with label -- output:", output, "label:", label)
    dim = len(output)
    
    for i in range(0, dim):
        if output[i] >= 0.5: # output[i] is set to 0
            if label[i] != 1:
                return False

    return True


file_path = "data\\test.txt"

data = pd.read_csv(file_path, sep=",", header=None)

list_X = np.stack([data[0].values, data[1].values, data[2].values, data[3].values], axis=1)
list_X = list_X.reshape(-1, 4)

#list_y = np.array(data[4])

list_y = []
for label in data[4]:
    if label == 0:
        list_y.append([0,0])
    else:
        list_y.append([1,1])

list_y = np.array(list_y)
list_y = list_y.reshape(-1, 2)

featureNum = len(list_X[0])
labelDim = len(list_y[0])

nn = FCNN()

layerOneNodeNum = 3
layerOne = FCLayer(False, layerOneNodeNum, featureNum,  0.05)
nn.addLayer(layerOne)

layerTwoNodeNum = labelDim
layerTwo = FCLayer(True, layerTwoNodeNum, layerOneNodeNum, 0.05)
nn.addLayer(layerTwo)

trainFCNN(list_X, list_y, nn, 10, 0.01)
predictFCNN(nn, list_X, list_y)

