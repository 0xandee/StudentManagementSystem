import numpy as np

def encode(y):
    cats = list(set(y))
    num_cols = len(cats)
    num_rows = len(y)
    Y = np.zeros((num_rows, num_cols))
    for i in range(num_rows):
        for j in range(num_cols):
            if (y[i] == cats[j]):
                Y[i][j] = 1
        
    return Y