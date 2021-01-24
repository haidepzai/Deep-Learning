import gzip
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder


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


X_test = open_images("../mnist/t10k-images-idx3-ubyte.gz").reshape(-1, 784)
y_test = open_labels("../mnist/t10k-labels-idx1-ubyte.gz")


class LogisticRegression(object):
    def __init__(self, lr = 0.00001):
        self.lr = lr
        self.w = None
        self.b = None

    def cost(self, pred, y):
        return -np.mean(y * np.log(pred) + \
                        (1 - y) * np.log(1 - pred), axis=1)

    def fit(self, X, y):
        if self.w is None:
            self.w = np.zeros((y.shape[0], X.shape[1]))
        if self.b is None:
            self.b = np.zeros(y.shape[0])

        pred = expit((X @ self.w.T) + self.b).T
        e = pred - y

        dw = (e @ X / X.shape[0])
        db = np.mean(e, axis=1)

        self.w = self.w - self.lr * dw
        self.b = self.b - self.lr * db

        print("Kosten: " + str(self.cost(pred, y)))

    def predict(self, X):
        return expit((X @ self.w.T) + self.b).T

model = LogisticRegression(lr = 0.00001)

for i in range(0, 150):
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    y_test_pred = np.argmax(y_test_pred, axis=0)
    print(np.mean(y_test_pred == y_test))