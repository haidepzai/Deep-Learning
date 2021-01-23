import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


# Vorstellung: MNIST-Daten!
# http://yann.lecun.com/exdb/mnist/

def open_images(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16) \
            .reshape(-1, 28, 28) \
            .astype(np.float32)


def open_labels(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)


# reshape(-1, 784) : so viele Zeilen wie nötig (60.000) mit 784 Spalten (28x28)
# Davor war es (60000, 28, 28), jetzt (60000, 784)
X_train = open_images("../mnist/train-images-idx3-ubyte.gz").reshape(-1, 784)
y_train = open_labels("../mnist/train-labels-idx1-ubyte.gz")

# True/False in die Form 0/1 (Wir wollen das Model trainieren, dass es 4 erkennt)
y_train = (y_train == 4).astype(np.float32)


# print(X_train)
# print(X_train.shape)
# print(y_train)
# exit()


def S(x):
    return expit(x)  # Fertige Sigmoid Funktion, die schon implementiert wurde
    # return 1 / (1 + np.exp(-x))


def f(w, b, x):
    return S(w @ x.T + b)


def J(w, b, x, y):
    return -np.mean(y * np.log(f(w, b, x)) + \
                    (1 - y) * np.log(1 - f(w, b, x)))


def J_ableitung_w(w, b, x, y):
    e = f(w, b, x) - y
    return np.mean(x.T * e, axis=1)


def J_ableitung_b(w, b, x, y):
    return np.mean(f(w, b, x) - y)


# print(X_train.shape)
# exit()
lr = 0.00001
w = np.zeros((1, 784))  # Gewichte initialisieren in der Form 1 Zeile und 784 Spalten
b = 1
for i in range(0, 1000):
    dw = J_ableitung_w(w, b, X_train, y_train)
    db = J_ableitung_b(w, b, X_train, y_train)

    w = w - lr * dw
    b = b - lr * db

    cost = J(w, b, X_train, y_train)
    print("Kosten: " + str(cost))
