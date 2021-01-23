import gzip
import numpy as np
import matplotlib.pyplot as plt


# Vorstellung: MNIST-Daten!
# http://yann.lecun.com/exdb/mnist/

def open_images(filename):
    with gzip.open(filename, "rb") as file:  # rb = read binary
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=16) \
            .reshape(-1, 28, 28) \
            .astype(np.float32)


def open_labels(filename):
    with gzip.open(filename, "rb") as file:
        data = file.read()
        return np.frombuffer(data, dtype=np.uint8, offset=8)


X_train = open_images("../mnist/train-images-idx3-ubyte.gz")  # die Bilder
print(X_train.shape)

print(X_train[0].shape)
plt.imshow(X_train[0])
plt.show()

y_train = open_labels("../mnist/train-labels-idx1-ubyte.gz")  # 0, 1, 2 ...
print(y_train.shape)
print(y_train[0])
