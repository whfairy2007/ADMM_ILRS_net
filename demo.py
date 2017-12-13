
# coding: utf-8

# In[6]:


# good results
import numpy as np
from scipy.sparse import random
from scipy.optimize import fmin_bfgs
from solver import Solver
from ADMM_ILRS_net import ADMM_IRLS_Net

if __name__ == "__main__":
    # data generation
    # correlated generator
    M = 50 # number of features
    N = 500 # number of samples
    Belta = np.zeros((M, 1))
    Belta[8,0] = -0.82
    Belta[16,0] = 0.51
    Belta[25,0] = 0.76
    Belta[41,0] = 0.41
    # correlated features
    Belta[9,0] = -0.47
    Belta[17,0] = 0.72
    Belta[26,0] = 0.31
    Belta[42,0] = 0.68
    Belta = np.asmatrix(Belta)

    X = np.random.randn(N,M) # design matrix
    X[:,[9]] = X[:,[8]] + 0.01*np.random.randn(N, 1);
    X[:,[17]] = X[:,[16]] + 0.01*np.random.randn(N, 1);
    X[:,[26]] = X[:,[25]] + 0.01*np.random.randn(N, 1);
    X[:,[42]] = X[:,[41]] + 0.01*np.random.randn(N, 1);


    sigma = 0 # variance of the noise
    y = np.sign(np.dot(X,Belta)+sigma*np.random.rand(N,1)) #labels with noise


    # inference
    #SGD
    #get_ipython().magic('load_ext autoreload')
    #get_ipython().magic('autoreload 2')


    data = {
    'X_train': X,
    'y_train': y,
    'X_val': X[400:],
    'y_val': y[400:],
    }
    model = ADMM_IRLS_Net()
    learning_rate = 1e-3
    solver = Solver(model, data,
                print_every=1, num_epochs=2, batch_size=N,
                update_rule='sgd',
                optim_config={
                  'learning_rate': learning_rate
                }
         )
    solver.train()



    # results
    Belta_3 = model.state_var['Belta_3']
    print(np.around(np.column_stack((Belta,Belta_3)),2))




