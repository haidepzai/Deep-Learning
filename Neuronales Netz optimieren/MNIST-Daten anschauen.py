import gzip
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
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

images = X_train.reshape(-1, 28, 28)
# Rollen wird das Bild verschoben (Zahlen im np.array wird verschoben)
images = np.roll(images, (3, 3), axis=(1, 2))
plt.imshow(images[0])
plt.show()