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


X_test = open_images("../mnist/t10k-images-idx3-ubyte.gz").reshape(-1, 784)
y_test = open_labels("../mnist/t10k-labels-idx1-ubyte.gz")


class NeuralNetwork(object):
    # Konstruktor
    def __init__(self):
        # mit pickle.load die bereits gespeicherten Gewichte laden
        with open("w0.p", "rb") as file:
            self.w0 = pickle.load(file)
        with open("w1.p", "rb") as file:
            self.w1 = pickle.load(file)

    def activation(self, x):
        return expit(x)  # Sigmoid Aktivierungsfunktion

    # (f-funktion)
    def predict(self, X):
        # Eingabe erste Ebene (Aktivierung mit Sigmoid Funktion)
        a0 = self.activation(self.w0 @ X.T)  # Matrizenmultiplikation aus logitische Regression
        # Eingabe zweite Ebene (Matrizenmultiplikation mit dem Ergebnis/Ausgang der zweiten Ebene)
        pred = self.activation(self.w1 @ a0)
        return pred


model = NeuralNetwork()
# print(model.w0.shape)
# print(model.w1.shape)
# 784 Eingänge und 100 Ausgänge (100 mittlere Knoten im Hidden Layer)
# Nächste Layer hat 10 Ausgänge und 100 Eingänge (vom Hidden Layer)

y_test_pred = model.predict(X_test / 255.)
print(y_test_pred)
# argmax = Höchste Zahl zurückgeben als Index (Ist dann die vorhergesagte Zahl)
y_test_pred = np.argmax(y_test_pred, axis=0)
print(y_test_pred)
print(np.mean(y_test_pred == y_test))
