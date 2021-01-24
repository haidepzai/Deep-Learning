import numpy as np
import pandas as pd

y_train = ["Köln", "Berlin", "München", "Hamburg"]
data = pd.get_dummies(y_train)
print(data.values)

df = pd.read_csv("autos_prepared.csv")
print(pd.get_dummies(df, columns = ["fuelType"]))