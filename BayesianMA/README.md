In the following section is reported a Python implementation of the Stochastic Search Variable Selection (SSVS) described in *Variable Selection by Gibbs Sampling* (Robert McCulloch & Edward George, 1993).

The core idea of the model algorithm is the following: similarly to other shrinkage methods the coefficients $\beta$ are set with a prior mean equal to 0, but for each $j$-th coefficient a latent variable $\gamma_j$ is introduced which affects the $\beta$ variance. Specifically:

$\beta_j \mid \gamma_j \sim \gamma_j \mathcal{N}(0, c_j^2 \tau_j^2) + (1 - \gamma_j) \mathcal{N}(0, \tau_j^2)$


$c_j$ and $\tau_j$ are hyperparameters related to the strictness of the selection algorithm.

$c_j$ is the ratio of heights between the slab (i.e., distribution when $\gamma_j = 1$) and spike (i.e., distribution when $\gamma_j = 0$) evaluated at 0.

$\tau_j$ is the prior variance of $\beta$ and is chosen according to the scale of the data. Indeed, a too restrictive $\tau_j$ might censor coefficient values in the estimation phase.

In the original paper the author shows a criterion based on the intersection between the spike and slab distributions to infer the following couples of parameters $(\sigma_{\beta_j}/\tau_j, c_j)$:

- (1, 5)
- (1, 10)
- (10, 100)
- (10, 100)
- (10, 500)

Therefore the choice of $\tau_j$ depends on the scale of the data making the algorithm not scale invariant.

The procedure can be a suitable alternative to OLS stepwise selection algorithm without requiring to compute model statistics (e.g., $R^2$, AIC, BIC) for a large subset of models and provides a measure of uncertainty related to the variables within the model. In credit risk it can be used to develop satellite models to link the Probability of Default to macroeconomic conditions and perform scenario stress test analysis.

## Model and Prior

$$
y = X \beta + \varepsilon, \qquad
\varepsilon \sim \mathcal{N}(0, \sigma^2 I_n)
$$

$$
\gamma_j \in \{0,1\}, \qquad
\gamma_j \sim \mathrm{Ber}(p_j)
$$

$$
\beta_j \mid \gamma_j \sim
\mathcal{N}(0, \gamma_j c^2 \tau^2 + (1-\gamma_j)\tau^2)
$$

In vector form:

$$
\beta \mid \gamma \sim \mathcal{N}(0, D_\gamma R D_\gamma^T)
$$

where $D_\gamma$ is defined as:

$$
D_\gamma =
\begin{pmatrix}
a_1 \tau_1 & 0 & \cdots & 0 \\
0 & a_2 \tau_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & a_k \tau_k
\end{pmatrix}
$$

with

$$
a_j =
\begin{cases}
c_j^2 & \text{if } \gamma_j = 1 \\
1 & \text{if } \gamma_j = 0
\end{cases}
$$

and

$$
R = \sigma^2 (X^T X)^{-1} \quad \text{or} \quad I
$$

$$
\sigma^2 \sim IG(a_0, b_0)
$$

- All models considered are linear Gaussian regressions under the homoskedasticity assumption.  
- $\gamma_j$ is a latent indicator variable that flags whether regressor $j$ is included in the model. In this application $p_j$ is set to 0.5 to let the data guide the selection of the variable.

## Gibbs Sampler Estimation

The algorithm is implemented through a Gibbs sampling strategy. Specifically, three full conditional distributions are derived:

- $\beta \mid y, \gamma, \sigma^2$  
- $\gamma \mid y, \sigma^2, \beta$  
- $\sigma^2 \mid y, \beta, \gamma$  

The three full conditional posterior distributions can be obtained analytically and the Gibbs sampler can be initialized by iteratively sampling from them.

### Full Conditional Posterior

The full conditional posterior distributions are the following.

$$
\beta \mid y, \gamma, \sigma^2 \sim \mathcal{N}(\beta^*, K^{-1})
$$

$$
\beta^* = K^{-1} X^T y / \sigma^2
$$

$$
K = X^T X / \sigma^2 + (D_\gamma R D_\gamma^T)^{-1}
$$

$$
\sigma^2 \mid y, \beta, \gamma \sim IG(a^*, b^*)
$$

$$
a^* = a_0 + n / 2
$$

$$
b^* = b_0 + (y - X \beta)^T (y - X \beta) / 2
$$

$$
P(\gamma_j = 1 \mid \cdot) = P(\gamma_j = 1 \mid \beta) =
\frac{p_j f(\beta_j \mid \gamma_j = 1)}
{p_j f(\beta_j \mid \gamma_j = 1) + (1 - p_j) f(\beta_j \mid \gamma_j = 0)}
$$
