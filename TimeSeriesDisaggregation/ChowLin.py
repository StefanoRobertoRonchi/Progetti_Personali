import os 
import pandas as pd
import numpy as np 
import sys 
import matplotlib.pyplot as plt
import scipy.optimize

def Chow_Lin(y_low,y_high, strategy):
    '''
    Breve funzione che applica il metodo di decomposizione delle serie storiche di Chow-Lin descritto in Best Linear Unbiased Interpolation,
    Distribution, and Extrapolation of Time Series by Related Series (1971) con 3 possibili strategie di ancoraggio:
    - distribution : Ancoraggio che la somma dei periodi di alta frequenza sia pari al valore di ancoraggio della serie a bassa frequenza (Es. le vendite annuali devono essere pari
    alla somma delle vendite trimestrali).
    - start: L'ancoraggio tra la serie ad alta frequenza e bassa frequenza è sulla prima osservazione del periodo (i.e., il valore annuale combacia con il primo valore della serie trimestrale/
    mensile.
    - end: L'ancoraggio avviene tra l'ultimo periodo della serie ad alta frequenza e il valore annuale (i.e., il valore annuale combacia con il valore dell'ultimo mese/trimestre)
    '''
    # Definizione della matrice di Conversione Low -> High Frequency
    nL = y_low.shape[0]   # low freq (anni)
    nH = y_high.shape[0]  # high freq (trimestri)
    C = np.zeros((nL, nH))
    if strategy == "distribution":
        step = (nH//nL)
        for i in range(nL):
            start = step*i
            stop = start + step
            C[i,start:stop] = 1/step
    if strategy == "start":
        step = (nH//nL)
        for i in range(nL):
            C[i,i*step] = 1
    if strategy == "end":
        step = (nH//nL)
        for i in range(nL):
            C[i,step*(i+1)-1] = 1 
            
    
    ############## Definizione e Starting Value ##############
    X_low = C @ (y_high.reshape(-1,1))
    y_low = y_low.reshape(-1,1)
    # Initial Guess for the Var Cov matrix (u_t = alpha * u_(t-1) + eps_t)
    # Define the Var-Cov Matrix of the errors
    V = lambda alpha, T, sigma2=1.0: (sigma2/(1 - alpha**2)) * alpha**np.abs(
    np.subtract.outer(np.arange(T), np.arange(T))) # np.subtract outer creates a matrix of differences 
    # the two vectors (in this case each row is the element of the first array less the elements of the second one)
    q = np.random.uniform(-1,1)
    tol = 1 # starting value for convergence tolerance
    
    al = lambda a : (C @ V(a,nH) @ C.T)
    equation = lambda a : (al(a)[0,1]/al(a)[0,0]) - q
    res = scipy.optimize.minimize_scalar(lambda a: equation(a)**2,
                      bounds=(-0.999, 0.999), method='bounded',
                      options={'xatol':1e-8})
    alpha_hat = float(res.x)
    max_iter = 1000
    it = 0
    
    ############## Calcolo dei Beta ##############
    def Beta_ChowLin(V,C,q,alpha_hat,X_low,y_low):
        # Define starting Low Frequency VarCov
        var_low = C @ V(alpha_hat,nH) @ C.T
        var_low = (var_low + var_low.T) * 0.5 
        # calcolo Beta tramite step con Cholesky
        L, lower = scipy.linalg.cho_factor(var_low) # decompongo VarCov in LL' = var_low
        SX =  scipy.linalg.cho_solve((L, lower), X_low)  # trovo inv(var_low) * X
        Sy =  scipy.linalg.cho_solve((L, lower), y_low)  # trovo inv(var_low) * y
        Beta_chow = np.linalg.solve(X_low.T @ SX, X_low.T @ Sy)
        return Beta_chow
    
    while (tol > 1e-4) and (it <= max_iter):
        Beta_chow = Beta_ChowLin(V,C,q,alpha_hat,X_low,y_low)
        # calcolo i residui low frequency
        res = y_low - X_low @ Beta_chow
        # calcolo il nuovo alpha
        res_mean = res.mean()
        r = res - res_mean
        res_var = (np.dot(r[:-1].ravel(), r[:-1].ravel()))
        alpha_tmp = np.dot(r[1:].ravel(), r[:-1].ravel()) / res_var
        tol = abs(alpha_tmp - alpha_hat)
        it+=1
        alpha_hat = alpha_tmp
    
    print(it)
    Beta_chow = Beta_ChowLin(V,C,q,alpha_hat,X_low,y_low)
    res = y_low - X_low @ Beta_chow
    var_low = C @ V(alpha_hat,nH) @ C.T
    ########## Calcolo degli shock ad alta frequenza ##########
    shock = V(alpha_hat,nH) @ C.T @ np.linalg.inv(var_low) @ res
    
    y_high_hat = y_high.reshape(-1,1) @ Beta_chow + shock
    return y_high_hat.ravel()
        
        
