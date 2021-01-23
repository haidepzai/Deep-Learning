import numpy as np
import matplotlib.pyplot as plt


def f(a, x):
    return a * x

# Sum Squared Error
def J(a, x, y):
    return (y - a * x) ** 2


def J_ableitung_a(a, x, y):
    return -2 * x * (y - a * x)


point1 = (1, 4)
point2 = (1.5, 5)

lr = 0.05
a = 1
for i in range(0, 50):
    da = J_ableitung_a(a, point1[0], point1[1])
    a = a - lr * da

    da = J_ableitung_a(a, point2[0], point2[1])
    a = a - lr * da

    cost = J(a, point1[0], point1[1]) + J(a, point2[0], point2[1])
    print("Kosten wenn a = " + str(a) + ": " + str(cost))

xs = np.arange(-2, 2, 0.1)
ys = f(a, xs)
plt.plot(xs, ys)

plt.scatter(point1[0], point1[1], color="red")
plt.scatter(point2[0], point2[1], color="green")
plt.show()

