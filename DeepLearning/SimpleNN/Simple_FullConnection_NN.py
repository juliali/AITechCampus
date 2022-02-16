import random
import numpy as np
import math
import pandas as pd


def convert_array_to_vertical_matrix(a):
    matrix = np.array(a).reshape(len(a),1)
    return matrix


def matrix_transfer(m):
    if len(m.shape) != 2:
        raise Exception("The matrix is not a 2dimension matrix.")
    crow = m.shape[0]
    ccol = m.shape[1]

    rm = np.array([0 for v in range(crow * ccol)]).reshape(ccol, crow)

    for i in range(0, crow):
        for j in range(0, ccol):
            rm[j][i] = m[i][j]

    return rm


def matrix_unit_substraction(m):
    if len(m.shape) != 2:
        raise Exception("The matrix is not a 2dimension matrix.")
    crow = m.shape[0]
    ccol = m.shape[1]

    rm = np.array([0 for v in range(crow * ccol)]).reshape(crow,ccol)

    for i in range(0, crow):
        for j in range(0, ccol):
            rm[i][j] = 1 - m[i][j]

    return rm

def matrix_summation(matrixA, matrixB):
    if len(matrixA.shape) != 2 or len(matrixB.shape) != 2:
        raise Exception("There is non-2dimension matrix.")

    if matrixA.shape[0] != matrixB.shape[0] or matrixA.shape[1] != matrixB.shape[1]:
        raise Exception("Matrixes are not in the same size do hadamard product.")

    row = matrixA.shape[0]
    col = matrixA.shape[1]

    matrixR = np.array([0 for v in range(row * col)]).reshape(row, col)

    for i in range(0, row):
        for j in range(0, col):
            matrixR[i][j] = matrixA[i][j] + matrixB[i][j]

    return matrixR


def matrix_subtraction(matrixA, matrixB):
    if len(matrixA.shape) != 2 or len(matrixB.shape) != 2:
        raise Exception("There is non-2dimension matrix.")

    if matrixA.shape[0] != matrixB.shape[0] or matrixA.shape[1] != matrixB.shape[1]:
        raise Exception("Matrixes are not in the same size do hadamard product.")

    row = matrixA.shape[0]
    col = matrixA.shape[1]

    matrixR = np.array([0 for v in range(row * col)]).reshape(row, col)

    for i in range(0, row):
        for j in range(0, col):
            matrixR[i][j] = matrixA[i][j] - matrixB[i][j]

    return matrixR


#def vector_dotproduct(vectorA, vectorB):
#    if len(vectorA) != len(vectorB):
#        raise Exception("Size is not same for the two vectors to do dot product.")
#    r = 0
#    for i in range(0, len(vectorA)):
#        r += vectorA[i] * vectorB[i]
#   return r
def matrix_times_scalar(scalar, m):
    if len(m.shape) != 2:
        raise Exception("The matrix is not a 2dimension matrix.")
    crow = m.shape[0]
    ccol = m.shape[1]

    rm = np.array([0 for v in range(crow * ccol)]).reshape(crow,ccol)

    for i in range(0, crow):
        for j in range(0, ccol):
            rm[i][j] = m[i][j] * scalar

    return rm


def matrix_multiplication(matrixA, matrixB):
    if len(matrixA.shape) != 2 or len(matrixB.shape) != 2:
        raise Exception("There is non-2dimension matrix.")

    if matrixA.shape[1] != matrixB.shape[0]:
        raise Exception("Matrixes are not matched to do multiplication.")

    rrowNum = matrixA.shape[0]
    rcolNum = matrixB.shape[1]
    matrixR = np.array([0 for v in range(rrowNum * rcolNum)]).reshape(rrowNum, rcolNum)

    for i in range(0, rrowNum):
        for j in range(0, rcolNum):
            for ci in range(0, len(matrixA[i])):
                matrixR[i][j] += matrixA[i][ci] * matrixB[ci][j]

    return matrixR


def matrix_hadamard_product(matrixA, matrixB):
    if len(matrixA.shape) != 2 or len(matrixB.shape) != 2:
        raise Exception("There is non-2dimension matrix.")

    if matrixA.shape[0] != matrixB.shape[0] or matrixA.shape[1] != matrixB.shape[1]:
        raise Exception("Matrixes are not in the same size do hadamard product.")

    row = matrixA.shape[0]
    col = matrixA.shape[1]

    matrixR = np.array([0 for v in range(row * col)]).reshape(row, col)

    for i in range(0, row):
        for j in range(0, col):
            matrixR[i][j] = matrixA[i][j] * matrixB[i][j]

    return matrixR


def matrix_sigmoid(m):
    if len(m.shape) != 2:
        raise Exception("The matrix is not a 2dimension matrix.")
    crow = m.shape[0]
    ccol = m.shape[1]

    rm = np.array([0 for v in range(crow * ccol)]).reshape(ccol, crow)

    for i in range(0, crow):
        for j in range(0, ccol):
            rm[i][j] = 1.0 / (1.0 + math.exp(0.0 - m[i][j]))

    return rm


class FCLayer():
    def __init__(self, isLastLayer, nodeNum, inputVectorSize, stepSize):
        self.stepSize = stepSize
        self.isLastLayer = isLastLayer
        self.nodeNum = nodeNum

        self.W = np.array([random.random() for v in range(inputVectorSize * nodeNum)]).reshape(nodeNum, inputVectorSize)
        self.B = np.array([random.random()] for v in range(nodeNum)).reshape(1, nodeNum)

        return


    def forward(self, inputData):
        Z = matrix_summation(matrix_multiplication(self.W, inputData), self.B)
        A = matrix_sigmoid(Z)
        return A


    def backword(self, inputData, dZNext, WNext):
        A = self.forword(inputData)
        A_T = matrix_transfer(A)

        if self.isLastLayer:
            Y_T = matrix_transfer(dZNext)
            dZ = matrix_subtraction(A_T, Y_T)
        else:

            IA = matrix_unit_substraction(A)
            IA_T = matrix_transfer(IA)

            dZ = matrix_multiplication(dZNext, WNext)
            dZ = matrix_hadamard_product(dZ, A_T)
            dZ = matrix_hadamard_product(dZ, IA_T)

        dW = matrix_multiplication(inputData, dZ)
        dB = dZ

        self.W = matrix_subtraction(self.W, matrix_times_scalar(self.stepSize, dW))
        self.B = matrix_subtraction(self.B, matrix_times_scalar(self.stepSize, dB))

        return dZ, self.W

class FCNN():
    def __init__(self):
        self.layerArray = []
        return

    def addLayer(self, aLayer):
        self.layerArray.append(aLayer)
        return

    def forword(self, inputData):
        print(inputData)
        currentInput = inputData
        for aLayer in self.layerArray:
            currentOutput = self.forword(currentInput)
            currentInput = currentOutput
        return currentOutput

    def backword(self, inputData, labelData, stepSize):
        layerNum = len(self.layerArray)
        dZNext = labelData
        WNext = None
        for i in range(layerNum - 1, -1, -1):
            aLayer = self.layerArray[i]
            dZNext, WNext = aLayer.backword(inputData, dZNext, WNext)

        return

    def countLoss(self, matrixA, matrixB):
        if len(matrixA.shape) != 2 or len(matrixB.shape) != 2:
            raise Exception("There is non-2dimension matrix.")

        if matrixA.shape[0] != matrixB.shape[0] or matrixA.shape[1] != matrixB.shape[1]:
            raise Exception("Matrixes are not matched to do multiplication.")

        crow = matrixA.shape[0]
        ccol = matrixB.shape[1]

        total = 0.0
        number = 0

        for i in range(0, crow):
            for j in range(0, ccol):
                a = matrixA[i][j]
                b = matrixB[i][j]
                currentLoss = -(a * math.log(b) - (1 - a) * math.log(1 - b))
                total += currentLoss
                number += 1

        loss = total/ float(number)

        return loss


def trainFCNN(inputDatas, labelDatas, maxEpochNum, threshold, stepSize):
    featureNum = len(inputDatas[0])
    #layerOne = FCLayer(False, 2, featureNum)
    layerTwo = FCLayer(True, featureNum, 1)

    nn = FCNN()

    #nn.addLayer(layerOne)
    nn.addLayer(layerTwo)

    loss = 100.0
    loopNum = 0

    while loss > threshold and loopNum < maxEpochNum:
        for inputData in inputDatas:
            output = nn.forword(inputData)
            nn.backword(stepSize)

        loss = nn.countLoss(labelDatas, output)
        loopNum += 1

    print("Final Loss:", loss)
    print("Output", output)

    return nn

def predictFCNN(nn, inputDatas, labelDatas):

    for inputData in inputDatas:
        output = nn.forword(inputData)
        print(output)

    loss = nn.countLoss(labelDatas, output)
    print(loss)
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

m3 = matrix_multiplication(m1, m2)
#print(m3)

m4 = convert_array_to_vertical_matrix(a1)
#print(m4)
m5 = matrix_transfer(m4)
#print(m5)


file_path = "data\\data_banknote_authentication.txt"

data = pd.read_csv(file_path, sep=",", header=None)

list_X = np.stack([data[0].values, data[1].values, data[2].values, data[3].values], axis=1)
list_X = list_X.reshape(-1, 4)
print(list_X)
list_y = data[4]

