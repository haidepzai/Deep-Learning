import numpy as np
import matplotlib.pyplot as plt


# Geradengleichung, a ist die Steigung (Kostengerade)
def f(a, x):
    return a * x


# Kostenfunktion in Abhängigkeit von a (Parabel)
# X und Y sind die Koordinaten von point (1, 4)
def J(a, x, y):
    return (y - a * x) ** 2


# Ableitung = Gerade (Ableitung = 0: Extrempunkt der Kostenfunktion)
def J_ableitung_a(a, x, y):
    return -2 * x * (y - a * x)


# x,y
point = (1, 4)
lr = 0.05
a = 1
# Kosten werden nach jeder Iteration minimiert
for i in range(0, 50):
    da = J_ableitung_a(a, point[0], point[1])  # Ableitung (Steigung)
    print("da = " + str(da))
    a = a - lr * da  # a nähert sich 4 (möglichst exakte Steigung finden, sodass Kosten (Ableitung) gegen 0 ist
    print("a = " + str(a))
    print("Kosten wenn a = " + str(a) + ": " + str(J(a, point[0], point[1])))

    xs = np.arange(-2, 2, 0.1)  # X Werte für die Gerade (Linie wird gezeichnet - Geradengleichung)
    ys = f(a, xs)
    plt.plot(xs, ys)

plt.scatter(point[0], point[1])
plt.show()
