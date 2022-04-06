
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.random import randn
import random
from scipy import signal
from scipy.sparse import rand
from CP import *
from DR import *
from Functions import J1,J2,J0, J0_1D,Jm2,l0_norm,l1_norm
from Utils import *

if __name__ == "__main__":
    v_t = [0.001, 0.01, 0.1, 1, 2]
    v_l = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]

    for t in v_t:
        
        eCP_1=[]; eCP_0=[]; eDR_0=[]; cCP_1=[]; cCP_0=[]; cDR_0=[];oCP_0=[];oDR_0=[]
        
        for l in v_l:
        
    #         solCP_1, errorsCP_1, corrCP_1, objectiveCP_1, resCP_1 = CP_l1(x_true, D, A, 10000, t=t, l=l ,rho=1, verbose=False)
            solCP_0, errorsCP_0, corrCP_0, objectiveCP_0, resCP_0 = CP_l0(x_true, D, A, 10000, t=t, l=l ,rho=1, verbose=False)
            solDR_0, errorsDR_0, corrDR_0, objectiveDR_0, resDR_0 = PD_DRS_l0(x_true, D, A,10000, t=t, l=l ,rho=1,verbose=False)
            
    #         eCP_1.append(errorsCP_1[-1])
            eCP_0.append(errorsCP_0[-1])
            eDR_0.append(errorsDR_0[-1])
    #         cCP_1.append(corrCP_1[-1])
            cCP_0.append(corrCP_0[-1])
            cDR_0.append(corrDR_0[-1])
            oCP_0.append(objectiveCP_0[-1])
            oDR_0.append(objectiveDR_0[-1])          
            
        print('t = ', t)
        fig, axs = plt.subplots(1,3, figsize=(15,5))
    #     axs[0].loglog(v_l,eCP_1,label='CP l1')
        axs[0].loglog(v_l,eCP_0,label='CP')
        axs[0].loglog(v_l,eDR_0,label='PD-DRS')
        axs[0].legend()
        axs[0].set_xlabel("Regularization parameter")
        axs[0].set_ylabel("Relative error")
    #     axs[1].semilogx(v_l,cCP_1,label='CP l1')
        axs[1].semilogx(v_l,cCP_0,label='CP')
        axs[1].semilogx(v_l,cDR_0,label='PD-DRS')
        axs[1].legend()
        axs[1].set_xlabel("Regularization parameter")
        axs[1].set_ylabel("Correlation")    
        axs[2].loglog(v_l,oCP_0,label='CP')
        axs[2].loglog(v_l,oDR_0,label='PD-DRS')
        axs[2].legend()
        axs[2].set_xlabel("Regularization parameter")
        axs[2].set_ylabel("Objective function")
        plt.show()
