import numpy as np
import matplotlib.pyplot as plt

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


def f(w, b, xs):
    rs = []  # Liste
    for x in xs:  # xs = X_train; x = [0.0, 0.0]
        sum = 0  # summe von w[0] * x[0] + w[1] * x[1] + ... + w[n] * x[n]
        for i in range(0, len(w)):  # len(w) in dem Fall 2
            sum = sum + w[i] * x[i]  # 1. Iteration ist 0 + 1 * 0
        r = S(sum + b)  # r = S(w[0] * x[0] + w[1] * x[1] + b)
        rs.append(r)  # in Liste einpacken
    return np.array(rs)


# Kostenfunktion (Kreuzentropie)
def J(w, b, x, y):
    return -np.mean(y * np.log(f(w, b, x)) + \
                    (1 - y) * np.log(1 - f(w, b, x)))


def J_ableitung_w(w, b, x, y):
    rs = []
    for i in range(0, len(w)): # len(w) in dem Fall 2
        rs.append(np.mean(x[:, i] * (f(w, b, x) - y)))
        # x[:, i] Alle Zeilen, i. Spalte (z.B. w0 ist dann 0, w1 ist 1 usw)
    return np.array(rs)

# return np.array([
#       J_ableitung_w0(w, b, x, y),
#       J_ableitung_w0(w, b, x, y)
# ])


def J_ableitung_b(w, b, x, y):
    return np.mean(f(w, b, x) - y)


# Input
lr = 0.1
w = np.array([1, 1])  # Gewichte
b = 1  # bias
for i in range(0, 1000):
    dw = J_ableitung_w(w, b, X_train, y_train)
    db = J_ableitung_b(w, b, X_train, y_train)

    w = w - lr * dw
    b = b - lr * db

    cost = J(w, b, X_train, y_train)
    print("Kosten: " + str(cost))

print("w0 = " + str(w[0]))
print("w1 = " + str(w[1]))
print("b = " + str(b))

# Testdaten
print(f(w, b, [
    [0, 0],
    [1, 1]
]))
