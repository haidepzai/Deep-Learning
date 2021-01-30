import gzip
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
import pickle


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
y_train_oh = oh.fit_transform(y_train.reshape(-1, 1)).toarray()

X_test = open_images("../mnist/t10k-images-idx3-ubyte.gz").reshape(-1, 784)
y_test = open_labels("../mnist/t10k-labels-idx1-ubyte.gz")

class NeuralNetwork(object):
    def __init__(self, lr = 0.1):
        self.lr = lr

        # Gewichte mit kleinen Zahlen initialisieren
        self.w0 = np.random.randn(100, 784)
        self.w1 = np.random.randn(10, 100)


    def activation(self, x):
        return expit(x)

    def train(self, X, y):
        # Eingabe erste Ebene (Aktivierung)
        a0 = self.activation(self.w0 @ X.T)  # Matrizenmultiplikation aus logitische Regression
        # Eingabe zweite Ebene (Matrizenmultiplikation mit dem Ergebnis/Ausgang der zweiten Ebene)
        pred = self.activation(self.w1 @ a0)

        # Soll - Ist
        e1 = y.T - pred # finaler error
        # Backpropagation error
        e0 = e1.T @ self.w1 # error aus der Ebene davor

        dw1 = e1 * pred * (1 - pred) @ a0.T / len(X)
        # Backpropagation Steigung
        dw0 = e0.T * a0 * (1 - a0) @ X / len(X) # X sind die Eingänge

        assert dw1.shape == self.w1.shape
        assert dw0.shape == self.w0.shape

        self.w1 = self.w1 + self.lr * dw1
        self.w0 = self.w0 + self.lr * dw0

        print("Kosten: " + str(self.cost(pred, y)))

    def predict(self, X):
        a0 = self.activation(self.w0 @ X.T)
        pred = self.activation(self.w1 @ a0)
        return pred

    def cost(self, pred, y):
        # SUM((y - pred)^2)
        s = (1 / 2) * (y.T - pred) ** 2
        return np.mean(np.sum(s, axis=0))

model = NeuralNetwork()

for i in range(0, 500):
    model.train(X_train / 255., y_train_oh)

    y_test_pred = model.predict(X_test / 255.)
    y_test_pred = np.argmax(y_test_pred, axis=0)
    print(np.mean(y_test_pred == y_test))