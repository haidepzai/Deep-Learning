import numpy as np
import matplotlib.pyplot as plt


# Geradengleichung, a ist die Steigung (Kostengerade)
# Quasi die Predict Funktion
def f(a, x):
    return a * x


# Sum Squared Error
# Kostenfunktion in Abhängigkeit von a (Parabel)
# X und Y sind die Koordinaten von point (1, 4)
def J(a, x, y):
    return (y - a * x) ** 2  # Soll - Ist im Quadrat


# Ableitung = Gerade (Ableitung = 0: Extrempunkt der Kostenfunktion)
# Kettenregel. Partielle Ableitung hinsichtlich a (a ist Variable, der Rest wie Zahlen)
def J_ableitung_a(a, x, y):
    return -2 * x * (y - a * x)


# x,y
point = (1, 4)
lr = 0.05
a = 1  # Steigung
# Kosten werden nach jeder Iteration minimiert
for i in range(0, 50):
    da = J_ableitung_a(a, point[0], point[1])  # Ableitung (Steigung) (nähert sich 0)
    print("da = " + str(da))
    a = a - lr * da  # a nähert sich 4 (möglichst exakte Steigung finden, sodass Kosten (Ableitung) gegen 0 ist
    print("a = " + str(a))
    print("Kosten wenn a = " + str(a) + ": " + str(J(a, point[0], point[1])))

    xs = np.arange(-2, 2, 0.1)  # X Werte für die Gerade (Linie wird gezeichnet - Geradengleichung)
    ys = f(a, xs)
    plt.plot(xs, ys)

plt.scatter(point[0], point[1])
plt.show()
