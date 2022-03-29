import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.random import randn
import random
from scipy import signal
from scipy.sparse import rand

random.seed(9999)
np.set_printoptions(precision=2)
# np.random.seed(9999)

##Generate the true signal
d, n = 100, 50
rs = 20
x_true = rand(d,1,0.1,random_state=rs)-2*rand(d,1,0.05,random_state=rs)
x_true = x_true.toarray()
x_true = np.cumsum(x_true,axis=0)
x_true = np.squeeze(np.asarray(x_true))
A = np.random.rand(n,d)

####Linear operator##############
D = np.eye(d)- np.vstack((np.zeros((1,d)),np.hstack((np.eye(d-1),np.zeros((d-1,1))))))
# D = D[1:,:]