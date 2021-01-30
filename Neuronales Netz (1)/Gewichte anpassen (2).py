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

        with open("w0.p", "rb") as file:
            self.w0 = pickle.load(file)

        self.w1 = np.zeros((10, 100))

    # Sigmoid Aktivierungsfunktion
    def activation(self, x):
        return expit(x)

    def train(self, X, y):
        # Eingabe erste Ebene
        a0 = self.activation(self.w0 @ X.T)  # Matrizenmultiplikation aus logitische Regression
        # Eingabe zweite Ebene (Matrizenmultiplikation mit dem Ergebnis/Ausgang der zweiten Ebene)
        pred = self.activation(self.w1 @ a0)

        e = y.T - pred # Fehler

        # Steigung
        dw1 = e * pred * (1 - pred) @ a0.T / len(X)

        assert dw1.shape == self.w1.shape # Sicherstellen, dass die shape übereinstimmen (gleiche Dimension)
        # Gewichte aktualisieren
        self.w1 = self.w1 + self.lr * dw1

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
    # trainieren
    model.train(X_train / 255., y_train_oh)

    y_test_pred = model.predict(X_test / 255.)
    y_test_pred = np.argmax(y_test_pred, axis=0)
    print(np.mean(y_test_pred == y_test))