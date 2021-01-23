import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x ** 2 - 4 * x + 5


def f_ableitung(x):
    return 2 * x - 4


x = 5
plt.scatter(x, f(x), c="r") # Plotten 1 Punkt (Scatter Plot) mit der Farbe rot
for i in range(0, 25):
    steigung_x = f_ableitung(x)
    x = x - 0.05 * steigung_x # 0.05 Schrittweite "Learning Rate"
    plt.scatter(x, f(x), c="r")
    print(x)

# numpy array (list)
xs = np.arange(-2, 6, 0.1) # Werte generieren von -2 bis 6 in 0.1 Schritte
ys = f(xs) # Auf alle X werte die Funktion f(x) anwenden
plt.plot(xs, ys) # X und Y Werte plotten und als Linie anzeigen
plt.show()