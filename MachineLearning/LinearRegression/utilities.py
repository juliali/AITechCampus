import pandas as pd
import matplotlib.pyplot as plt


def draw_graph(X, y, y_pred):
    # 将测试结果以图标的方式显示出来
    plt.scatter(X, y, color='black')
    plt.plot(X, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()


def dump(a, b, path):
    with open(path, "w") as file:
        file.write("a,b" + "\n")
        file.write(str(a) + ',' +  str(b))
    return


def load(path):
    model = pd.read_csv(path)
    a = model['a']
    b = model['b']

    return a, b

def predict(path, X):
    a, b = load(path)
    predicted_y = []
    for x in X:
        y = a + b * x
        predicted_y.append(y)

    return predicted_y