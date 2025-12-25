In the following section is reported the Python implementation of the Stochastic Search Variable Selection (SSVS) described in  _Variable Selection by Gibbs Sampling_ (Robert McCulloch & Edward George,1993).

The model,implemented via Python is the following:

## Model

$$
y = X\beta + \varepsilon, 
\qquad 
\varepsilon \sim \mathcal{N}(0, \sigma^2 I_n)
$$

$$
\gamma_j \in \{0,1\}
$$

- (1) All models considered are linear Gaussian regressions under the homoskedasticity assumption.  
- (2) $\gamma_j$ is a latent indicator variable that flags whether regressor $j$ is included in the model.

## Prior

For each regressor $j$, the prior distribution of its associated coefficient $\beta_j$ is:

$$
\beta_j \mid \gamma_j \sim 
\mathcal{N}\!\left(
0,\;
\gamma_j \tau^2 + (1-\gamma_j)c^2\tau^2
\right)
$$
