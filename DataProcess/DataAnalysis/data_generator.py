from numpy import *
from random import *
import matplotlib.pyplot as plt


def loadData():
    x = arange(-1, 1, 0.02)
    y = ((x * x - 1) ** 3 + 1) * (cos(x * 2) + 0.6 * sin(x * 1.3))
    # 生成的曲线上的各个点偏移一下，并放入到xa,ya中去
    xr = [];
    yr = [];
    i = 0
    for xx in x:
        yy = y[i]
        d = float(randint(80, 120)) / 100
        i += 1
        xr.append(xx * d)
        yr.append(yy * d)
    return x, y, xr, yr


def figPlot(x1, y1, x2, y2):
    plt.plot(x1, y1, color='g', linestyle='-', marker='')
    plt.plot(x2, y2, color='m', linestyle='_', marker='.')
    plt.show()


def main():
    x, y, xr, yr = loadData()
    figPlot(x, y, xr, yr)