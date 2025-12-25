In the following section is reported the Python implementation of the Stochastic Search Variable Selection (SSVS) described in  _Variable Selection by Gibbs Sampling_ (Robert McCulloch & Edward George,1993).

The model,implemented via Python is the following:

$$
The Model:
y = X\beta + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,\sigma^2 I_n). (1)
\gamma_j \in {0,1} (2) 
$$ 

 (1) All the models considered are linear gaussian regressions with homoskedasticity assumption.
 (2) is a latent variable which "flags" whether the variable is selected or not.
$$
Prior:
for each variable j, the Prior distribution of its related coefficient \beta_j is:
f(\beta | \gamma_j) = \mathcal{N}(0,\gamma_j * \tau^2 + (1-\gamma_j)*c^2*\tau^2
$$
