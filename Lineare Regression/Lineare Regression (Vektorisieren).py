import numpy as np
import matplotlib.pyplot as plt


def f(a, x):
    return a * x


def J(a, x, y):
    return np.mean((y - a * x) ** 2)


def J_ableitung_a(a, x, y):
    return np.mean(-2 * x * (y - a * x))


points = np.array([
    [1, 4],
    [1.5, 5],
    [2, 8],
    [0.5, 3]
])

lr = 0.05
a = 1
for i in range(0, 50):
    da = J_ableitung_a(a, points[:, 0], points[:, 1])
    a = a - lr * da

    # [:, 0] Alle Zeilen, 1 Spalte
    cost = J(a, points[:, 0], points[:, 1])
    print("Kosten wenn a = " + str(a) + ": " + str(cost))

xs = np.arange(-2, 2, 0.1)
ys = f(a, xs)
plt.plot(xs, ys)

plt.scatter(points[:, 0], points[:, 1], color="red")
plt.show()

