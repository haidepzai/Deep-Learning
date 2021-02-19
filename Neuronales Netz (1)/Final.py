import gzip
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
import pickle


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


X_train = open_images("../mnist/train-images-idx3-ubyte.gz").reshape(-1, 784)
y_train = open_labels("../mnist/train-labels-idx1-ubyte.gz")

X_test = open_images("../mnist/t10k-images-idx3-ubyte.gz").reshape(-1, 784)
y_test = open_labels("../mnist/t10k-labels-idx1-ubyte.gz")


class NeuralNetwork(object):
    # Konstruktor
    def __init__(self):
        self.w0 = np.random.randn(100, 784)
        self.w1 = np.random.randn(10, 100)
        # Gewichte mit kleinen Zahlen initialisieren

    # Sigmoid Aktivierungsfunktion
    def activation(self, x):
        return expit(x)

    def train(self, X, y):
        # Eingabe erste Ebene
        a0 = self.activation(self.w0 @ X.T)  # Matrizenmultiplikation aus logitische Regression
        # Eingabe zweite Ebene (Matrizenmultiplikation mit dem Ergebnis/Ausgang der zweiten Ebene)
        pred = self.activation(self.w1 @ a0)

        output_error = y.T - pred  # True - Pred
        dw1 = output_error * pred * (1 - pred) @ a0.T / len(X)

        # print(np.mean(dw1))

        a0_error = output_error.T @ self.w1
        dw0 = (a0_error.T * a0 * (1 - a0)) @ X / len(X)

        # Sicherstellen, dass die shape übereinstimmen (gleiche Dimension)
        assert dw1.shape == self.w1.shape
        assert dw0.shape == self.w0.shape

        # Gewichte anpassen
        self.w1 += 0.1 * dw1
        self.w0 += 0.1 * dw0

    # (f-funktion)
    def predict(self, X):
        # Eingabe erste Ebene (Aktivierung mit Sigmoid Funktion)
        a0 = self.activation(self.w0 @ X.T)  # Matrizenmultiplikation aus logitische Regression
        # Eingabe zweite Ebene (Matrizenmultiplikation mit dem Ergebnis/Ausgang der zweiten Ebene)
        pred = self.activation(self.w1 @ a0)
        return pred

    def cost(self, pred, y):
        # SUM((y - pred)^2)
        s = (1 / 2) * (y.T - pred) ** 2
        return np.mean(np.sum(s, axis=0))


model = NeuralNetwork()
# print(model.w0.shape)
# print(model.w1.shape)

# Kostenfunktion
oh = OneHotEncoder()
y_train_oh = oh.fit_transform(y_train.reshape(-1, 1)).toarray()

# print(y_train)
# print(y_train_oh)

# Nicht 60.000 Datensätze trainieren, sondern mit Schrittweite von 1000
for i in range(0, 1000):
    for j in range(0, 59000, 1000):
        model.train(X_train[j:(j + 1000), :] / 255., y_train_oh[j:(j + 1000), :])

    y_test_pred = model.predict(X_test / 255.)
    y_test_pred = np.argmax(y_test_pred, axis=0)
    print(np.mean(y_test_pred == y_test))
