import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Praxisbeispiel: Preis von Autos bestimmen!
#   -> Partielle Ableitung
#   -> Bias

df = pd.read_csv("./autos_prepared.csv")

points = df[["kilometer", "price"]].values

# Skalieren
s = StandardScaler()
points = s.fit_transform(points)

def f(a, b, x):
    return a * x + b


def J(a, b, x, y):
    return np.mean((y - (a * x + b)) ** 2)

# Partielle Ableitung hinsichtlich a (a ist dann eine Variable)
def J_ableitung_a(a, b, x, y):
    return np.mean(-2 * x * (-a * x - b + y))

# Partielle Ableitung hinsichtlich b
def J_ableitung_b(a, b, x, y):
    return np.mean(((-2) * (((-a) * x) - b + y)))


lr = 0.005
a = 1
b = 1
for i in range(0, 500):
    da = J_ableitung_a(a, b, points[:, 0], points[:, 1])
    db = J_ableitung_b(a, b, points[:, 0], points[:, 1])
    a = a - lr * da
    b = b - lr * db

    cost = J(a, b, points[:, 0], points[:, 1])
    print("Kosten wenn a = " + str(a) + ": " + str(cost))

xs = np.arange(-4, 4, 0.5)
ys = f(a, b, xs)

line_points = np.c_[xs, ys]
line_points = s.inverse_transform(line_points)
plt.plot(line_points[:, 0], line_points[:, 1])

points_orig = s.inverse_transform(points)

plt.scatter(points_orig[:, 0], points_orig[:, 1])
plt.show()