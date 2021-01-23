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
    1.0,
    0.0,
    1.0
])


def S(x):
    return 1 / (1 + np.exp(-x))


def f(w, b, x):
    return S(w @ x.T + b) # @ = Matrizenmultipliktion (wie man es kennt)


def J(w, b, x, y):
    return -np.mean(y * np.log(f(w, b, x)) + \
                    (1 - y) * np.log(1 - f(w, b, x)))

def J_ableitung_w(w, b, x, y):
    e = f(w, b, x) - y
    return np.mean(x.T * e, axis=1) # * = element wise multiplication; axis = 1 : Zeilenweise

#w = np.array([[1, 2]])
#b = 3
#x = np.array([
#    [0, 0],
#    [0, 1],
#    [1, 0],
#    [1, 1]
#])
#y = np.array([0, 1, 0, 1])
# [ 0.24488529 -0.00229137]
#print(J_ableitung_w(w, b, x, y))
#exit()

def J_ableitung_b(w, b, x, y):
    return np.mean(f(w, b, x) - y)


lr = 0.1
w = np.array([[1, 1]])
b = 1
for i in range(0, 1000):

    dw = J_ableitung_w(w, b, X_train, y_train)
    db = J_ableitung_b(w, b, X_train, y_train)

    w = w - lr * dw
    b = b - lr * db

    cost = J(w, b, X_train, y_train)
    print("Kosten: " + str(cost))

print("w0 = " + str(w[0, 0]))
print("w1 = " + str(w[0, 1]))
print("b = " + str(b))

# Testdaten
print(f(w, b, np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])))