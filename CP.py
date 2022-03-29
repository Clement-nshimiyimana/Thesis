import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.random import randn
import random
from scipy import signal
from scipy.sparse import rand

def CP_l0(x_true, D, A,maxit, t=0.1, l=1,rho=1, verbose=True):
    """Chambolle pock algorithm 0-norm primal and dual    
    t= primal stepsize
    l= regularization parameter
    rho = relaxation parameter
    x_true: input signal
    D: Sparse matrix
    A: recovering matrix
    s: dual stepsize
    """
    s=1/(t*norm(D)**2)
    
    num_sample, num_feat = D.shape
    b = A @ x_true

    INV=inv(np.eye(num_feat)+t*A.T @ A)

    r = t*INV @ A.T @ b
    
    x = np.random.rand(num_feat)
    y = np.random.rand(num_sample)
    
    objective = []
    errors = []
    corr = []
    res = []
    
    if verbose:
        print("Lauching CP 0-norm solver...")
        print(' | '.join([name.center(8) for name in ["it", "err", "obj"]]))
        
    for k in range(maxit):
        
        rho=1/np.log(k+2) 
        
        xold = np.copy(x)
        yold = np.copy(y)
        x = J2(INV,r, xold - t*D.T @ y)
        temp = (y+s*D @(2*x-xold))
        y = temp-s*J0(l/s,temp/s)
        
        x= xold+rho*(x-xold)
        y= yold+rho*(y-yold)
        
        error = norm(x_true - x)/ norm(x_true)
        co2D = np.corrcoef(x_true, x)
        co = co2D[0,1]
        ff = 1/2*norm(A @ x-b)**2+l*l0_norm(D @ x)
        errors.append(error)
        objective.append(ff)
        corr.append(co)
        res.append(norm(x-xold))
        

        if k%(maxit/10)==0 and verbose:
            print(' | '.join([("%d" % k).rjust(8), 
#                                   ("%.2e" % y).rjust(8),
                              ("%.2e" % error).rjust(8),
                              ("%.2e" % ff).rjust(8)]))       
    return x, errors, corr, objective, res

