import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.random import randn
import random
from scipy import signal
from scipy.sparse import rand

def PD_DRS_l0(x_true, D, A,maxit, t=0.1, l=1,rho=1,verbose=True):
    """Primal-Dual DRS with L0-norm regularization    
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
    a = np.hstack((np.eye(num_feat), t*D.T))
    c = np.hstack((-s*D , np.eye(num_sample)))
    Inversa=inv(np.vstack((a,c)))
    Fx_DR = []
    
    q = np.random.rand(num_sample)
    p = np.random.rand(num_feat) 
    
    objective = []
    objective1 = []
    errors = []
    corr = []
    res = []
    
    if verbose:
        print("Lauching DR 0-norm solver...")
        print(' | '.join([name.center(8) for name in ["iteration", "err", "obj"]]))      
    for k in range(maxit):
          
        rho=1/np.log(k+2) 
        
        pold = np.copy(p)
        
        x = J2(INV,r, p)

        z = q-s*J0(l/s,q/s)####
      
        m = np.resize(np.vstack((2*x-p, 2*z-q)),Inversa.shape[1])
        vect = Inversa@(m)
        u = vect[:num_feat]

        v = vect[num_feat:]

        p = p+rho*(u-x)
        q = q+rho*(v-z)
        qq = np.resize(q,x_true.shape)
        error = norm(x_true - p)/ norm(x_true)
        ff = 1/2*norm(A @ p-b)**2+l*l0_norm(D @ p) 
        
        co2D = np.corrcoef(x_true, x)
        co = co2D[0,1]
        corr.append(co)
        errors.append(error)
        objective.append(ff)
        res.append(norm(p-pold))
        
        if k%(maxit/10)==0 and verbose:
            print(' | '.join([("%d" % k).rjust(8), 
                              ("%.2e" % error).rjust(8),
                              ("%.2e" % ff).rjust(8)])) 
            
    return x, errors, corr, objective, res


def CP_l1(x_true, D, A,maxit, t=0.1, l=1,rho=1, verbose=True):
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

    #x = np.random.rand(num_feat,1)
    #y = np.random.rand(num_sample,1)
    
    x = np.random.rand(num_feat)
    y = np.random.rand(num_sample)
    
    objective = []
    errors = []
    corr = []
    res = []
    
    if verbose:
        print("Lauching CP 1-norm solver...")
        print(' | '.join([name.center(8) for name in ["it", "err", "obj"]]))
        
    for k in range(maxit):
        
        rho=1/np.log(k+2) 
        
        xold = np.copy(x)
        yold = np.copy(y)
        x = J2(INV,r, xold - t*D.T @ y)
        temp = (y+s*D @(2*x-xold))
        y = temp-s*J1(l/s,temp/s)
        
        x= xold+rho*(x-xold)
        y= yold+rho*(y-yold)
        
        error = norm(x_true - x)/ norm(x_true)
        co2D = np.corrcoef(x_true, x)
        co = co2D[0,1]
        ff = 1/2*norm(A @ x-b)**2+l*l1_norm(D @ x)
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