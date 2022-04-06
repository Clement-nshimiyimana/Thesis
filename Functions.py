import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.random import randn
import random
from scipy import signal
from scipy.sparse import rand

def Jm2(par, x):
    y = (1+par/norm(x))*x
    return y

###linearity#####
def J2(INV, r, x):
    #r = INV*sigma*A.T * b
    y = INV @ x + r
#     print('inv:',INV.shape)
#     print('x',x.shape)
    return y

# L1 norm
def J1(par,x):
    return np.sign(x)*np.maximum(np.abs(x)-par,0) 

#L0 norm 
def J0(par,x):
    d = len(x)
    y = np.zeros(d)
    for i in range(d):
        y[i] = J0_1D(par,x[i])###
    return y

def J0_1D(par,x):
    if abs(x) > np.sqrt(2*par):
        out = x
    else:
        out = 0
    return out


def l0_norm(x):
    count = []
    for i in x:
        if i !=0:
            count.append(i)
        else:pass
    return len(count)

def l1_norm(x):
    return norm(x, ord=1)