import numpy as np
import matplotlib.pyplot as plt


# Volljährigkeit vorhersagen!
# Spalte 1: Alter X_Train
# Spalte 2: Volljährig: Ja / Nein? Y_train
points = np.array([
    [20, 1],
    [17, 0],
    [15, 0],
    [10, 0],
    [30, 1],
    [40, 1],
    [35, 1],
    [13, 0],
    [5, 0],
    [18, 1],
    [25, 1],
    [8, 0]
])

# Sigmoid Funktion
def S(x):
    return 1 / (1 + np.exp(-x))

# Verwendet die Sigmoid Funktion (> 0.5 = volljährig; < 0.5 = minderjährig)
def f(a, b, x):
    return S(a * x + b)

# Kreuzentropie (Average Cross Entropy Cost)
# cost = -(1.0/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1-A))
def J(a, b, x, y):
    return -np.mean(y * np.log(f(a, b, x)) + (1 - y) * np.log(1 - f(a, b, x)))


def J_ableitung_a(a, b, x, y):
    return np.mean(x * (S(a * x + b) - y))


def J_ableitung_b(a, b, x, y):
    return np.mean(S(a * x + b) - y)


lr = 0.05
a = 1
b = 1
for i in range(0, 10000):
    da = J_ableitung_a(a, b, points[:, 0], points[:, 1])
    db = J_ableitung_b(a, b, points[:, 0], points[:, 1])
    a = a - lr * da
    b = b - lr * db

    cost = J(a, b, points[:, 0], points[:, 1])
    print("Kosten: " + str(cost))

print("Neue Parameter: " + str(a) + " und " + str(b))
xs = np.arange(1, 60, 0.5)
ys = f(a, b, xs)
plt.plot(xs, ys)

plt.scatter(points[:, 0], points[:, 1], c="r")
plt.show()

# Anwenden mit Testdaten
print("----")
print(f(a, b, np.array([16, 22])))
print(f(a, b, 18))