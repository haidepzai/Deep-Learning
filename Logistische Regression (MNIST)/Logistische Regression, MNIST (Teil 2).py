import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit


# Vorstellung: MNIST-Daten!
# http://yann.lecun.com/exdb/mnist/

def open_images(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16)\
            .reshape(-1, 28, 28)\
            .astype(np.float32)


def open_labels(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)


X_train = open_images("../mnist/train-images-idx3-ubyte.gz").reshape(-1, 784)
y_train = open_labels("../mnist/train-labels-idx1-ubyte.gz")
print(y_train)
y_train = (y_train == 4).astype(np.float32)
print(y_train)

print("----------")

X_test = open_images("../mnist/t10k-images-idx3-ubyte.gz").reshape(-1, 784)
y_test = open_labels("../mnist/t10k-labels-idx1-ubyte.gz")
print(y_test)
y_test = (y_test == 4).astype(np.float32)
print(y_test)


def S(x):
    return expit(x)
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


lr = 0.00001
w = np.zeros((1, 784))
b = 1
for i in range(0, 5):

    dw = J_ableitung_w(w, b, X_train, y_train)
    db = J_ableitung_b(w, b, X_train, y_train)

    w = w - lr * dw
    b = b - lr * db

    cost = J(w, b, X_train, y_train)
    print("Kosten: " + str(cost))


y_test_pred = f(w, b, X_test) > 0.5
y_test_pred = y_test_pred.reshape(-1)

print(y_test_pred) # Gelernten Daten mit den Testdaten X
print(y_test) # Hier sind die Testdaten (True/False; wenn 4 dann True)
# Vergleichen
print(np.mean(y_test == y_test_pred))