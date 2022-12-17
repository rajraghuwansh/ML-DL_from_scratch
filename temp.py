import math
import numpy as np
def sig(x):
    y =  (1 + math.exp(-x))**-1
    z = 1/(1 + np.exp(-x))
    return y,z

if __name__ == "__main__":
    x = int(input())
    print(sig(x))