import numpy as np
from scipy.special import expit

def S(x):
    return expit(x)


wx = np.array([
    [1, 2, 3, 4, 5],
    [-1, -2, -3, -4, -5]
])
b = np.array([1, -1])

print(wx.T)
f = S(wx.T + b).T


y = np.array([
    [1, 0],
    [1, 0],
    [0, 1],
    [1, 0],
    [1, 0]
])

e = f - y.T
print(e)
