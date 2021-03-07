import numpy as np

def softened_sigmoid(x, T=1):
    ss = 1 / (1 + np.exp(-x/T))
    return ss

if __name__ == '__main__':
    values = np.array([-100, -10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10, 100])
    print(values)
    print(softened_sigmoid(values))
    print(softened_sigmoid(values, 10))