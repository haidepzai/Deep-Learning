import numpy as np

y_train = np.array([1, 5, 8, 4, 2, 3, 4, 5, 6, 7, 3, 8, 9, 0]).reshape(-1, 1)

from sklearn.preprocessing import OneHotEncoder

oh = OneHotEncoder()
oh.fit(y_train)

print(oh.transform(y_train).toarray())