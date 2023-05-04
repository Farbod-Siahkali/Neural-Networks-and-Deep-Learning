import numpy as np

def activation(x):
    teta = 1
    x[x >= 1] = 1
    x[x <1 ] = 0
    return x

def mult(a1, a0, b1, b0):
    x = np.array([a0, a1, b0, b1])
    x = x.reshape(4, 1)
    w1 = np.array( [[2/3,  0,  2/3,  0],
                    [0,   2/3, 2/3,  0],
                    [2/7, 2/7, 2/7, 2/7]])
    x2 = np.dot(w1, x)
    res = activation(x2)
    res = np.zeros([4,1]);
    res[3] = x2[0]
    res[2] = 0.75*a0 + 0.75*b1 + 1.5*x2[1] - 2.5*x2[2]
    res[1] = 2/3*a1 + 2/3*b1 - 2/3*x2[2]
    res[0] = x2[2]
    res = activation(res)
    return res.reshape(1,4)

print(mult(0, 0, 0, 0))
print(mult(0, 0, 0, 1))
print(mult(0, 0, 1, 1))
print(mult(0, 1, 0, 1))
print(mult(1, 0, 1, 0))
print(mult(1, 0, 0, 1))
print(mult(1, 0, 1, 1))
print(mult(1, 1, 1, 1))