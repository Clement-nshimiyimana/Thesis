
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
from Functions import *
from Utils import *


# solCP_1, errorsCP_1, corrCP_1, objectiveCP_1, resCP_1 = CP_l1(x_true, D, A, 10000, t=0.1, l=1 ,rho=1, verbose=True)
solCP_0, errorsCP_0, corrCP_0, objectiveCP_0, resCP_0 = CP_l0(x_true, D, A, 10000, t=0.1, l=1,rho=1, verbose=True)
solDR_0, errorsDR_0, corrDR_0, objectiveDR_0, resDR_0 = PD_DRS_l0(x_true, D, A,10000, t=0.1, l=1,rho=1,verbose=True)


fig, axs = plt.subplots(1,2, figsize=(15,5))
# axs[0,0].plot(x_true)
# axs[0,0].set_title('True signal')
# axs[0,1].plot(solCP_1)
# axs[0,1].set_title('CP for l1-norm')
axs[0].plot(x_true,label='Original Signal')
axs[0].plot(solCP_0,label='Produced Signal')
axs[0].set_title('CP')
axs[0].legend()
axs[1].plot(x_true,label='Original Signal')
axs[1].plot(solDR_0, label='Produced Signal')
axs[1].set_title('PD-DRS')
axs[1].legend()
plt.subplots_adjust(left=0.1,
                    bottom=0.3, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
plt.show()

fig, axs = plt.subplots(1,2, figsize=(15,5))
# axs[0].semilogy(errorsCP_1,label='CP l1')
axs[0].semilogy(errorsCP_0,label='CP')
axs[0].semilogy(errorsDR_0,label='PD-DRS')
axs[0].legend()
axs[0].set_title('Relative errors')
# axs[1].plot(corrCP_1,label='CP l1')
axs[1].plot(corrCP_0,label='CP')
axs[1].plot(corrDR_0,label='PD-DRS')
axs[1].legend()
axs[1].set_title('Correlation')
plt.show()


plt.semilogy(objectiveCP_0,label='CP')
plt.semilogy(objectiveDR_0,label='PD-DRS')
plt.legend()
plt.title('Objective function')
plt.show()