import numpy as np

a = np.array([1, 2, 3])

# Pad
print(np.pad(a, ((2, 2)), mode="constant"))
print(np.pad(a, ((1, 3)), mode="constant"))

# Roll
print(np.roll(a, (1, )))

