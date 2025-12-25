In the following section is reported a Python implementation of the Stochastic Search Variable Selection (SSVS) described in *Variable Selection by Gibbs Sampling* (Robert McCulloch & Edward George, 1993).

The core idea of the model algorithm is the following: similarly to other shrinkage methods the coefficients $\beta$ are set with a prior mean equal to 0, but for each $j^{th}$ coefficient a latent variable $\gamma_j$ is introduced which affects the $\beta$ variance. Specifically:

$$
\beta_j \mid \gamma_j \sim 
\gamma_j \, \mathcal{N}\!\left(0, c_j^2 \tau_j^2\right)
+ (1 - \gamma_j)\, \mathcal{N}\!\left(0, \tau_j^2\right)
$$

$c_j$ and $\tau_j$ are hyperparameters related to the strictness of the selection algorithm.

$c_j$ is the ratio of heights between the slab (i.e., distribution when $\gamma_j = 1$) and spike (i.e., distribution when $\gamma_j = 0$) evaluated at 0 and can be interpreted as the **prior odds** that the $j^{th}$ regressor should be included or not.

$\tau_j$ is the prior variance of $\beta$ and is chosen according to the scale of the data. Indeed, a too restrictive $\tau_j$ might censor coefficient values in the estimation phase.

In the original paper the author shows a criterion based on the intersection between the spike and slab distributions to infer the following couples of parameters $(\sigma_{\beta_j}/\tau_j, c_j)$:

- (1, 5)
- (1, 10)
- (10, 100)
- (10, 100)
- (10, 500)

Therefore the choice of $\tau_j$ depends on the scale of the data making the algorithm **not** scale invariant.

The procedure can be a suitable alternative to OLS stepwise selection algorithm without requiring to compute model statistics (e.g., $R^2$, AIC, BIC) for a large subset of models and provides a measure of **uncertainty related to the variables within the model**. In credit risk can be used to develop **satellite model to link the Probability of Default to macroeconomic conditions** and make scenario stress test analysis.

## Model and Prior

$$
y = X\beta + \varepsilon, 
\qquad 
\varepsilon \sim \mathcal{N}(0, \sigma^2 I_n)
\qquad (1)
$$

$$
\gamma_j \in \{0,1\} \qquad  \gamma_j \sim \mathrm{Ber}(p_j)
\qquad (2)
$$

$$
\beta_j \mid \gamma_j \sim 
\mathcal{N}\!\left(
0,\;
\gamma_j \tau^2 + (1-\gamma_j)c^2\tau^2
\right)
$$

In vector form:

$$
\beta \mid \gamma \sim \mathcal{N}\!\left(0,\; D_{\gamma} R D_{\gamma}^{\top}\right)
$$

with $D_{\gamma}$ defined as

$$
D_{\Gamma} =
\operatorname{diag}\!\left(
a_1 \tau_1^2,\ldots,a_k \tau_k^2
\right),
\qquad
a_j =
\begin{cases}
c_j^2, & \gamma_j = 1, \\
1,     & \gamma_j = 0.
\end{cases}
\qquad
R = \sigma^2( X'X)^{-1} \quad \text{or} \quad I
$$

$$
\sigma^2 \mid \gamma \sim \mathcal{IG}(a_0,b_0)
$$

- (1) All models considered are linear Gaussian regressions under the homoskedasticity assumption.  
- (2) $\gamma_j$ is a latent indicator variable that flags whether regressor $j$ is included in the model. In this application $p_j$ is set to 0.5 to let the data guide the selection of the variable (i.e., indifference prior).

## Gibbs Sampler Estimation

The algorithm is implemented through a Gibbs sampling strategy. Specifically three Gibbs sampler are derived:

- $\beta \mid y, \gamma, \sigma^2$  (3)  
- $\gamma \mid y, \sigma^2, \beta$  (4)  
- $\sigma^2 \mid y, \beta, \gamma$  (5)

The three full conditional posterior can be obtained analytically and then the Gibbs sampler can be initialized sampling (3), (4) and (5) iteratively.

### Full Conditional Posterior

The three full conditional posterior are the following:

$$
\begin{aligned}
\beta \mid y,\gamma,\sigma^2 
&\sim \mathcal{N}\!\left(\beta^*,\, K^{-1}\right),\\
\beta^* 
&= K^{-1}\frac{X^{\top}y}{\sigma^2},\\
K 
&= \frac{X^{\top}X}{\sigma^2} + \left(D_\gamma R D_\gamma^{\top}\right)^{-1}
\end{aligned}
$$

$$
\begin{aligned}
\sigma^2 \mid y,\beta,\gamma &\sim \mathcal{IG}(a^*, b^*),\\
a^* &= a_0 + \frac{n}{2},\\
b^* &= b_0 + \frac{1}{2}(y - X\beta)^{\top}(y - X\beta)
\end{aligned}
$$

$$
P(\gamma_j = 1 \mid \beta, \gamma_{-j}, \sigma^2)
=
\frac{PDF_{\beta_j, \gamma_j=1}}
{PDF_{\beta_j, \gamma_j=1} + PDF_{\beta_j, \gamma_j=0}}
$$
