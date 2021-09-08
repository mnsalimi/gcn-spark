import numpy as np 
from numpy import linalg as LA

def norm(M1, M2):
    C = np.subtract(M1, M2)
    norm1 = LA.norm(C)
    maxx = np.maximum(M1, M2)
    norm2 = LA.norm(maxx)
    return float(norm1/norm2)

sampl1 = np.random.uniform(low=-1, high=1, size=(10, 5))
sampl2 = np.random.uniform(low=-1, high=1, size=(10, 5))
# sampl1 = [
#     [1, 2, 3, 4],
#     [1, 2, 3, 4],
#     [4, 5, 1, 99],
# ]
# sampl2 = [
#     [1, 5, 3, 4],
#     [1, 2, 3, 4],
#     [12, 34, 67, 99]
# ]

print(norm(sampl1, sampl2))