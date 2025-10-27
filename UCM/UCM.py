import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import diags, eye, csc_matrix
from scipy.sparse.linalg import spsolve 
from scipy.linalg import solve_triangular

##### Data Import #####
gdp = pd.read_excel("GDP_IT.xlsx", parse_dates=["date"])
gdp = gdp.set_index("date")  # Italian Quarterly GDP from Q1 1981 to Q4 2019 
# Growth is not linear, transform in log since the model is additive.
gdp = np.log(gdp)*100


##### MCMC Initialization and Prior Definition #####
nsim = 4000
burnin = 500 
T = gdp.shape[0]
y = gdp["GDP"].values

## Tau0 ~ N(a0,b0) ##
# Trend Prior with initial value for Tau and Mu (Trend Growth), in the estimation are equivalent given as starting value Tau0 and Tau-1
# as prior mean I will use the first value of the TS
a0 = np.array([gdp.iloc[0], gdp.iloc[0]])
b0 = np.eye(2)*100
## Prior on the Cyclical coefficents Phi ~ N(phi0,Kp**-1) ##
phi0 = np.array([0.9,0.1]).reshape(-1,1)
Kp = np.eye(2) # Written as the precision Matrix (i.e., inverse of Variance-Covariance Matrix)

## Innovations of Cycle and Trend component ~ IG(a,b) ##
# Cycle : a=3 and b = 0.0002 
nu_c = 3
sigc2_prior = 2 # Average of the distribution (b/a-1) = 1 --> sigma = 1%. The cycle has innovation error ~ 1%
# Sigma Trend 
a_tau = 3    
b_tau = 0.006

##### Modelling System Equations ####
# Constructing the H matrix for the tau equation H*tau= Eta - 2 tau0 + tau-1
H = np.eye(T) 
H[np.arange(1,T), np.arange(0,T-1)] -=2 
H[np.arange(2,T), np.arange(0,T-2)] +=1
# Constructing H for phi  H_phi*cycle = Eps
H_phi = lambda phi1, phi2 : np.eye(T,T) - phi1*np.eye(T,T, k=-1) - phi2*np.eye(T,T, k=-2)
H_phi0 = H_phi(phi0[0,0], phi0[1,0])
# Constructing vector for how the starting trend enters in each equation
Xtau0 = np.column_stack((np.arange(2, T+2), -np.arange(1, T+1)))

##### Initialize MCMC #####
tau0 = np.array([gdp.iloc[0], gdp.iloc[0]]).reshape(-1,1)
sigc2 = .5
sigtau2= .001
phi = phi0.copy()
#### Doing some multiplication to speed up the sampler ####
# Matrix of the I(2) trend
H2H2 = H.T @  H
y_col = y.reshape(-1,1) # Reshaping the target vector


def UCM_sampler(y_col,nsim,burnin,
                tau0,Xtau0,a0,b0,
                phi,sigc2,H_phi,Kp,
                sigtau2,H2H2,a_tau,b_tau, rng = None):
    """
        Descrizione da fare....
    """
    if rng is None: # Setting the seed for the UCM chain
        rng = np.random.default_rng(104)
    
    T = len(y_col)
    b = np.zeros((T,1))
    ## Storing Values Array
    store_theta = np.zeros((nsim,6))# phi1, phi2, sigc2, sigtau2, Tau0m Tau-1
    store_tau = np.zeros((nsim,T))# Trend
    store_mu = np.zeros((nsim,T)) # Trend Growth
    
    for i in range(nsim+burnin):
        b[0,0] = 2*tau0[0] - tau0[1]
        b[1,0] = - tau0[0]
        H_phi_tmp = H_phi(phi[0,0], phi[1,0])
        ##### Sample Tau #####
        a = np.linalg.solve(H, b) 
        Ktau = ((H2H2)/sigtau2) + (H_phi_tmp).T @ H_phi_tmp/sigc2 
        ltau = (H_phi_tmp).T @ H_phi_tmp @ y_col/sigc2  + (H2H2 @ a) / sigtau2
        L_tau = np.linalg.cholesky(Ktau)
        tau_hat = np.linalg.solve(Ktau, ltau)
        z = rng.standard_normal((T,1))
        tau_draw = tau_hat + solve_triangular(L_tau.T, z, lower = False, check_finite = False)
        ##### Sample Phi #####
        c = y_col - tau_draw

        ## Creo la design matrix per la regressione del ciclo ##
        x1 = np.vstack([np.zeros((1, 1)), c[:-1]])   
        x2 = np.vstack([np.zeros((2, 1)), c[:-2]])   
        Xphi = np.hstack([x1, x2])  
        Kphi = Kp + (Xphi.T @ Xphi)/sigc2
        lphi = ((Kp @ phi0)+ (Xphi.T @ c)/sigc2)
        phi_hat = np.linalg.solve(Kphi, lphi)

        ## Check Stazionarietà ##
        statcond=0
        counter=-1
        while statcond == 0:
            counter+=1
            L_phi = np.linalg.cholesky(Kphi)
            z_phi = rng.standard_normal((2, 1))
            phi_draw = phi_hat + np.linalg.solve(L_phi.T, z_phi)
            ## Check Stationarity of the roots of phi_draw
            phi1, phi2 = float(phi_draw[0, 0]), float(phi_draw[1, 0])
            r = np.roots([-phi2, -phi1, 1])
            if counter == 100:
                print("error")
            if np.min(np.abs(r)) > 1.0:
                statcond = True
                phi = phi_draw.copy()

        ##### Sample sigc2 #####
        res = (c - Xphi @ phi).T @ (c - Xphi @ phi)
        SSE = 0.5*float(res)
        sigc2 = 1/rng.gamma(nu_c + T/2, 1/((sigc2_prior + SSE)))

        ## sample sigtau2

        del_tau = np.vstack([tau0[0:1], tau_draw]) - np.vstack([tau0[1:2], tau0[0:1], tau_draw[:-1]]) # calcolo delta Tau

        d = del_tau[1:] - del_tau[:-1]          # Δ^2 τ
        S_tau = float(d.T @ d)                  # somma dei quadrati (scalare)

        post_shape = a_tau + T/2.0
        post_rate  = b_tau + 0.5 * S_tau        # "rate" b della IG(a,b)

        sigtau2 = 1.0 / rng.gamma(post_shape, 1.0 / post_rate)

        ## Sample the initial condition tau0
        Ktau0 = np.linalg.inv(b0) + Xtau0.T @ H2H2 @ Xtau0/ sigtau2   
        l_tau0 = (np.linalg.inv(b0) @ a0) + Xtau0.T @ H2H2 @ tau_draw / sigtau2
        tau0_hat = np.linalg.solve(Ktau0, l_tau0)
        cF_tau0, lower_tau0 = cho_factor(Ktau0, lower=True, check_finite=False)
        z0 = rng.standard_normal((2, 1))
        tau0 = tau0_hat + solve_triangular(cF_tau0.T, z0,lower = False ,check_finite=False)

        if i % 1000 == 0:
            print(f"{i} iteration...")                    
        ##### Store se i è sopra le iter di burn_in #####
        if i > burnin:
            idx = i - burnin 
            store_tau[idx,:] =  tau_draw.ravel() # Trend
            store_theta[idx, :] =  np.hstack([phi.ravel(), sigc2, sigtau2, tau0.ravel()]) # phi1, phi2, sigc2, sigtau2, Tau0m Tau-1
            mu = np.empty((T, 1))
            mu[0, 0] = 4.0 * (tau_draw[0, 0] - tau0[0, 0])
            mu[1:, 0] = 4.0 * (tau_draw[1:, 0] - tau_draw[:-1, 0])
            store_mu[idx, :] = mu.ravel()
    return store_tau, store_mu, store_theta
