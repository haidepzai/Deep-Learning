import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.misc import imread
from sklearn.preprocessing import OneHotEncoder

# pillow
image = imread("2.png")
image = 255. - np.mean(image, axis=2).reshape(1, -1)

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

oh = OneHotEncoder()
y_train = oh.fit_transform(y_train.reshape(-1, 1)).toarray().T


# X_test = open_images("../mnist/t10k-images-idx3-ubyte.gz").reshape(-1, 784)
# y_test = open_labels("../mnist/t10k-labels-idx1-ubyte.gz")


def S(x):
    return expit(x)
    # return 1 / (1 + np.exp(-x))


def f(w, b, x):
    a = w @ x.T
    return S(a.T + b).T


def J(w, b, x, y):
    return -np.mean(y * np.log(f(w, b, x)) + \
                    (1 - y) * np.log(1 - f(w, b, x)), axis=1)


def J_ableitung_w(w, b, x, y):
    e = f(w, b, x) - y
    return (x.T @ e.T / x.shape[0]).T


def J_ableitung_b(w, b, x, y):
    return np.mean(f(w, b, x) - y, axis=1)


lr = 0.00001
w = np.zeros((10, 784)) # Gewichte mit 10 Zeilen unr 784 Spalten (28x28)
b = np.ones(10) # Zeilenvektor mit 10 Einträgen

# plt.imshow(X_train[0].reshape(28, 28))
# plt.show()

# plt.imshow(image.reshape(28, 28))
# plt.show()



for i in range(0, 100):

    dw = J_ableitung_w(w, b, X_train, y_train)
    db = J_ableitung_b(w, b, X_train, y_train)

    w = w - lr * dw
    b = b - lr * db

    cost = J(w, b, X_train, y_train)
    print("Kosten: " + str(cost))


    y_test_pred = f(w, b, image)
    y_test_pred = np.argmax(y_test_pred, axis=0)

    print(y_test_pred)

