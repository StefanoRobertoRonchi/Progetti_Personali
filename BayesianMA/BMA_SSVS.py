import os 
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm


################################################
# In the following script it's possible to find:
# (1) BMA estimation functions : Bayesian_MA_SSVS (estimation algorithm), 
# (2) PiP (Posterior inclusion Probability) extractor: PiP
# (3) Best Models and performance: TopModel
################################################

################### Funzioni ################### 
def Bayesian_MA_SSVS(y,X,n,burn_in,a=None,b=None,seed=42, identity = False, normalize_x = True,
                    normalize_y = False):
    """
    #### Hyperparameters priors and stating points ####
    ### Priors on Beta | Gamma ~ N_k(0, D_gamma*R*D_gamma) 
           where D = diag(a1*tau_1 , .... , ak*tau_k) where aj = 1 if gamma_j = 1 else aj = c
    ### Sigma^2 ~ IG(a,b) with avg as res_avg^2 OLS --> Don't depend on the specific model
    ### Gamma j ~ Bernoulli(pj) with pj = 0.5 --> flat prior for X_j to appear in the model
    """
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = np.asarray(y).ravel()
    else:
        y = np.asarray(y).ravel()
    
    # X: ensure 2D numpy array
    if isinstance(X, (pd.Series, pd.DataFrame)):
        X = X.values
    else:
        X = np.asarray(X)
    
    #____________________________________________________________
    #### Normalizzazione dati se attiva (consigliata)
    
    if normalize_x:
        mu_X = np.mean(X,axis=0)
        sigma_X = np.std(X, ddof=1,axis=0)
        ### Non standardizzo prima colonna che deve essere intercetta
        sigma_X[0] = 1
        mu_X[0] = 0
        X = (X - mu_X)/sigma_X
    if normalize_y:
        mu_y = np.mean(y,axis=0)
        sigma_y = np.std(y, ddof=1,axis=0)
        y = (y - mu_y)/sigma_y
    
    
    #____________________________________________________________
    #### Setting parametri di partenza
    X_t = X.T
    XtX = X.T @ X
    Xty = X.T @ y
    R0 = np.linalg.inv(XtX) 
    n_var = X.shape[1]
    rng = np.random.default_rng(seed=seed)
    
    #### Creo vettori di store 
    gamma_final = np.zeros((n-burn_in, n_var))
    betas_final = np.zeros((n-burn_in,n_var))
    sig_final = np.zeros((n-burn_in,1))
    gammas = []
    
    #### Definisco Starting Value e Prior se non definite in input
    gammas = np.ones(n_var)
    ### Beta starting value S
    Beta_OLS = np.linalg.solve(XtX, Xty) 
    sigma_OLS = np.sqrt(np.sum((y - X @ Beta_OLS)**2)/ (X.shape[0] - n_var))
    var_beta_ols = (sigma_OLS**2)* R0
    ### Sigma^2 starting value
    sigma_sq = sigma_OLS**2
    
    if identity: # ( I or (X'X)^-1)
        R = np.eye(n_var) 
    else:
        sX = np.sqrt(np.diag(R0))
        R  = R0 / np.outer(sX, sX) # Beta Correlation Matrix in the BMA model 

    
    if (a is None) or (b is None): # Sigma IG prior parameters
        sigma_2_avg = sigma_OLS**2
        a = 6
        b = (a - 1)*sigma_2_avg
        
    ### Slab and spike parameter     
    tau_sigma = 20 # scale parameter to shrink the starting value of the hyperparameter Tau (George McCulloch : rule of thumb = 10)
    c = tau_sigma*10 # Standard Dev. slab multiplicative factor
    tau_sq = (np.diag(var_beta_ols)/(tau_sigma**2)).ravel()  ## --> shrinkage parameter (Tau squared shrinked)
    for i in range(n):
        if i % 500 == 0:
            print(f"Iteration number {i}")
        #____________________________________________________________
        #### Estraggo i betas ~ N(Beta_post, V_beta_post)
        # Variance Beta structure 
        a_beta_var = 1 + (c -1)*gammas  # Slab variance indicator, if the indicator gamma = 1
        d = a_beta_var*np.sqrt(tau_sq) # beta prior variance is amplified by c^2
        D = np.diag(d)
        V_beta_post = np.linalg.inv(np.linalg.inv(D @ R @ D) + (X_t @ X)/sigma_sq)
        Beta_post = V_beta_post @ X_t @ y / sigma_sq
        # Sampling from a multivariate Normal
        Betas = Beta_post + np.linalg.cholesky(V_beta_post) @ rng.standard_normal(size = X.shape[1])
        #____________________________________________________________

        #### Estraggo le probabilità di vedere modello 
        unif = rng.random(n_var) # probability selection threshold
        # Gamma posterior : f(gamma =1)* f(beta| gamma =1) /(f(gamma =1)* f(beta| gamma =1) + f(gamma =0)* f(beta| gamma =0))
        sd_slab  = np.sqrt(tau_sq)*c
        sd_spike = np.sqrt(tau_sq)

        log_num = norm.logpdf(Betas, loc=0.0, scale=sd_slab)
        log_den = np.logaddexp(log_num, norm.logpdf(Betas, loc=0.0, scale=sd_spike))

        p_gamma1 = np.exp(log_num - log_den)
        gammas[1:] = (unif[1:] < p_gamma1[1:]).astype(int)
        gammas[0] = 1 # In ogni modello voglio l'intercetta

        #____________________________________________________________
        #### Estraggo sigma^2 posterior
        rss = (y - X @ Betas).T @ (y - X @ Betas)
        sigma_sq = 1/rng.gamma(a + y.size/2, 1/(b + rss/2))



        #____________________________________________________________
        #### Saving the posterior parameters
        if i >= (burn_in):
            idx = i - burn_in
            gamma_final[idx,] = gammas
            betas_final[idx,] = Betas
            sig_final[idx] = sigma_sq
    
    if normalize_x and normalize_y:
        betas_final[:,0] = mu_y + betas_final[:,0]*sigma_y - betas_final[:,1:] @ (mu_X[1:]*sigma_y/sigma_X[1:])
        betas_final[:,1:] = betas_final[:,1:]*sigma_y/sigma_X[1:]
        sig_final = sig_final*(sigma_y**2)
        output = {"Model_Number":np.arange(n-burn_in), 
              "Variable_Selected": gamma_final,
              "Betas": betas_final, 
              "Model Variance": sig_final}
    elif normalize_x:
        betas_final[:,0] = betas_final[:,0] - betas_final[:,1:] @ (mu_X[1:]/sigma_X[1:])
        betas_final[:,1:] = betas_final[:,1:]/sigma_X[1:]
        output = {"Model_Number":np.arange(n-burn_in), 
              "Variable_Selected": gamma_final,
              "Betas": betas_final, 
              "Model Variance": sig_final}
    elif normalize_y:
        betas_final[:,0] = betas_final[:,0]*sigma_y + mu_y
        betas_final[:,1:] = betas_final[:,1:]*sigma_y
        sig_final = sig_final*(sigma_y**2)
        output = {"Model_Number":np.arange(n-burn_in), 
              "Variable_Selected": gamma_final,
              "Betas": betas_final, 
              "Model Variance": sig_final}
    else:
        output = {"Model_Number":np.arange(n-burn_in), 
              "Variable_Selected": gamma_final,
              "Betas": betas_final, 
              "Model Variance": sig_final}

    return output

def PiP(BMA,X):
    """
    Extract the Posterior Inclusion Probability of each regressor. The output is a pandas Series.
    """
    if isinstance(X,pd.DataFrame):
        regressors = X.columns
    else:
        regressors = range(X.shape[1])

    PIP = np.mean(BMA["Variable_Selected"], axis = 0) # Posterior Inclusion Probability
    return pd.Series(PIP, index = regressors)


def TopModels(BMA,y,X, n=5, q_low = 0.25, q_high = 0.75):
    """
    Extract the list of most frequent models within the SSVS chain. For these promising combinations the R2 are computed for each model and 
    aggregated to report the p25,median,mean,p95.
    X must be the same used during the estimation. 
    
    """
    lista_modelli = pd.DataFrame(BMA["Variable_Selected"])
    beta_modelli = pd.DataFrame(BMA["Betas"])
    if isinstance(X,pd.DataFrame):
        lista_modelli.columns = X.columns
        beta_modelli.columns = X.columns
    else:
        lista_modelli.columns = range(X.shape[1])
        beta_modelli.columns = range(X.shape[1])
        
    lista_modelli.index = [f"Modello{i}" for i in range(1, lista_modelli.shape[0] + 1)]
    beta_modelli.index = [f"Modello{i}" for i in range(1, beta_modelli.shape[0] + 1)]
    
    #### Calcolo indicatori di fitting
    TSS = np.sum((y-y.mean())**2)
    for index,row in beta_modelli.iterrows():
        RSS = (y - X @ row).T @ (y - X @ row)
        R2 = 1 - (RSS/TSS)
        lista_modelli.loc[index,"R2"] = R2
        
    #### Salvataggio dei modelli 
    gamma_cols = [c for c in lista_modelli.columns if c != "R2"]

    model_selected = (
    lista_modelli
      .groupby(gamma_cols, dropna=False)["R2"]
      .agg(
          R2_mean="mean",
          R2_median="median",
          R2_q05=lambda s: s.quantile(0.05),
          R2_q95=lambda s: s.quantile(0.95),
          count="size"
      )
      .sort_values("count", ascending=False)
      .head(n)
      .reset_index()
        )
    return model_selected

################### Esempio ################### 

# Data test
y_test = np.zeros(1000)
X_test = np.zeros((1000,7))

np.random.seed(2)
n = 7
A = np.random.randn(n, n)
Sigma = A @ A.T                  # covariance PD
D = np.sqrt(np.diag(Sigma))
R = Sigma / np.outer(D, D)       # correlation matrix
L = np.linalg.cholesky(R)
sigma_eps = 0.2

for i in range(1000):
    X_test[i,:] = L @ np.random.standard_normal(size = 7)
    y_test[i] = 0.2*X_test[i,0] + .3*X_test[i,2] + 0.5*X_test[i,6] + sigma_eps*np.random.standard_normal(size = 1)
    
print(pd.DataFrame(X_test).corr())

models_test = Bayesian_MA_SSVS(y=y_test,X=X_test,n=10000,burn_in=500,a=None,b=None,seed=42, identity = False)
print(PiP(models_test,X_test))
example = TopModels(models_test,y_test,X_test)
example.columns = ["1","2","3","4","5","6","7","R2_mean","R2_median","R2_q05","R2_q95","count"]
print(example)
