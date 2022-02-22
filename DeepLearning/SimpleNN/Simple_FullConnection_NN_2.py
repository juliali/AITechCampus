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
        self.B = np.array([random.random() for v in range(nodeNum * 1)])#.reshape(nodeNum,1)
        print("-- self.W：", self.W)
        #print("initialized B:", self.B)

        return

    def forward(self, inputData):
        #print("Layer-forward, inputData:",inputData, "W:", self.W, "B:", self.B)
        M = np.dot(self.W, inputData)
        #print("M:", M)
        Z = np.add(M, self.B)
        A = expit(Z)
        return A

    def backward(self, inputData, dZNext, WNext):
        #print("Layer-backward, inputData:", inputData)

        A = self.forward(inputData)

        if self.isLastLayer:
            Y = dZNext
            dZ = np.subtract(A, Y)
            print("A:", A, "Y:", Y, "(1) dZ:", dZ)
        else:
            I = np.ones(len(A))
            IA = np.subtract(I, A)

            dZ = np.dot(dZNext, WNext)
            dZ = np.multiply(dZ, A)
            dZ = np.multiply(dZ, IA)
            print("(2) dZ:", dZ)

        #print("inputData:", inputData, "dZ:", dZ)
        dW = np.multiply(inputData, dZ)
        dB = dZ
        print("self.stepSize:", self.stepSize, "dW:", dW)
        self.W = np.subtract(self.W, np.multiply(self.stepSize, dW))
        print("self.W", self.W)
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
        #print("FCNN-forward, inputData:", inputData)
        currentInput = inputData
        for aLayer in self.layerArray:
            currentOutput = aLayer.forward(currentInput)
            currentInput = currentOutput
        return currentOutput

    def backward(self, inputData, labelData):
        #print("FCNN-backward, inputData:", inputData)
        layerNum = len(self.layerArray)
        dZNext = labelData
        WNext = None
        for i in range(layerNum - 1, -1, -1):
            aLayer = self.layerArray[i]
            dZNext, WNext = aLayer.backward(inputData, dZNext, WNext)

        return

    def countLoss(self, outputs, labels):
        #print("matrixA:", matrixA)
        #print("matrixB:", matrixB)

        crow = outputs.shape[0]
        ccol = outputs.shape[1]

        total = 0.0
        number = 0

        for i in range(0, crow):
            for j in range(0, ccol):
                a = outputs[i][j]
                b = labels[i][j]
                #print("a:", a, "b:", b)
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


def trainFCNN(inputDatas, labelDatas, maxEpochNum, threshold, stepSize):
    featureNum = len(inputDatas[0])
    #layerOne = FCLayer(False, 2, featureNum, stepSize)
    layerTwo = FCLayer(True, 1, featureNum, stepSize)

    nn = FCNN()

    #nn.addLayer(layerOne)
    nn.addLayer(layerTwo)

    loss = 100.0
    loopNum = 0

    while loss > threshold and loopNum < maxEpochNum:
        outputs = []
        index = 0
        #for inputData in inputDatas:
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

    return nn


def predictFCNN(nn, inputDatas, labelDatas):
    outputs = []
    caseNum = len(inputDatas)
    index = 0
    #for inputData in inputDatas:
    correctNum = 0
    while index < caseNum:
        output = nn.forward(inputDatas[index])
        outputs.append(output)
        if output > 0.5:
            output = 1
        else:
            output = 0

        if output == labelDatas[index][0]:
            #print("input:", inputDatas[index], "output:", output, "expected:", labelDatas[index][0], "True")
            correctNum += 1
        #else:
            #print("input:", inputDatas[index], "output:", output, "expected:", labelDatas[index][0], "False")

        index += 1


    outputs = np.array(outputs)
    loss = nn.countLoss(labelDatas, outputs)
    accurancy = correctNum * 1.0 / caseNum * 1.0 * 100.0
    print("Total Loss:", loss, "Accurancy:", str(accurancy) + "%")




    return


#layer = FCLayer(1, 2, 3, False, None, None)
a1 = [1,2,3,4,5,6]
a2 = [-1,-2,-3,-4,-5,-6]
#a3 = matrix_summation(a1, a2)
#a4 = matrix_subtraction(a3, a1)

#print(a3)
#print(a4)

m1 = np.array(a1).reshape(2,3)
m2 = np.array(a2).reshape(3,2)

#print(m1)
#print(m2)

#m3 = matrix_multiplication(m1, m2)
#print(m3)

#m4 = convert_array_to_vertical_matrix(a1)
#print(m4)
#m5 = matrix_transfer(m4)
#print(m5)


file_path = "data\\test.txt"

data = pd.read_csv(file_path, sep=",", header=None)

list_X = np.stack([data[0].values, data[1].values, data[2].values, data[3].values], axis=1)
list_X = list_X.reshape(-1, 4)
#print(list_X)
list_y = np.array(data[4])
list_y = list_y.reshape(-1, 1)

#print(list_y)

#list_X = np.array([[3.6216,8.6661,-2.8073,-0.44699],[2.2612,9.6101,-3.2037,-0.3988]])


#list_y = np.array([[0],[1]])

#print(list_X)
#print(list_y)

nn = trainFCNN(list_X, list_y, 10, 0.01, 0.05)
predictFCNN(nn, list_X, list_y)

