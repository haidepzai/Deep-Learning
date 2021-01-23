import numpy as np
import matplotlib.pyplot as plt

# Wahrheitstabelle
X_train = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

y_train = np.array([
    0.0,
    0.0,
    0.0,
    1.0
])


def S(x):
    return 1 / (1 + np.exp(-x))

# Aktivierungsfunktion
def f(w0, w1, b, x0, x1):
    return S(w0 * x0 + w1 * x1 + b)


def J(w0, w1, b, x0, x1, y):
    return -np.mean(y * np.log(f(w0, w1, b, x0, x1)) + \
                    (1 - y) * np.log(1 - f(w0, w1, b, x0, x1)))


def J_ableitung_w0(w0, w1, b, x0, x1, y):
    return np.mean(x0 * (f(w0, w1, b, x0, x1) - y))


def J_ableitung_w1(w0, w1, b, x0, x1, y):
    return np.mean(x1 * (f(w0, w1, b, x0, x1) - y))


def J_ableitung_b(w0, w1, b, x0, x1, y):
    return np.mean(f(w0, w1, b, x0, x1) - y)


lr = 0.1
w0 = 1
w1 = 1
b = 1
for i in range(0, 1000):
    dw0 = J_ableitung_w0(w0, w1, b, X_train[:, 0], X_train[:, 1], y_train)
    dw1 = J_ableitung_w1(w0, w1, b, X_train[:, 0], X_train[:, 1], y_train)
    db = J_ableitung_b(w0, w1, b, X_train[:, 0], X_train[:, 1], y_train)

    # Gewichte anpassen/aktualisieren
    w0 = w0 - lr * dw0
    w1 = w1 - lr * dw1
    b = b - lr * db

    cost = J(w0, w1, b, X_train[:, 0], X_train[:, 1], y_train)
    print("Kosten: " + str(cost))

print("w0 = " + str(w0))
print("w1 = " + str(w1))
print("b = " + str(b))
# True and True
print(f(w0, w1, b, 1.0, 1.0))

print(f(w0, w1, b, 1.0, 0.0))
print(f(w0, w1, b, 0.0, 0.0))

# xs = np.arange(1, 60, 0.5)
# ys = f(a, b, xs)
# plt.plot(xs, ys)

# plt.scatter(points[:, 0], points[:, 1], c="r")
# plt.show()

# print("----")
# print(f(a, b, np.array([16, 22])))
