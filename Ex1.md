This simulation example demonstrates the use of the `MLBC` package for
correcting bias and performing valid inference in regression models with
generated binary labels.

The example is based on the simulation design in [Battaglia,
Christensen, Hansen & Sacher (2024)](https://arxiv.org/abs/2402.15585).
Data are generated according to the model
`Y = b0 + b1 * X + (a1 X + a0 (1 - X)) * u`, where `u` is a standard
normal random variable. Parameter values are set to match the empirical
example in the paper.

In the main sample, the true variable `X` is latent. A predicted label
`Xhat` is generated with a false positive rate `fpr`.

We also generate a smaller validation sample in which both `X` and
`Xhat` are observed. This sample is used to estimate `fpr`.

We generate `nsim` data sets, each with `n` observations in the main
sample and `m` observations from which to estimate `fpr`.

Load the package:

``` r
library(MLBC)
```

Set parameter values and pre-allocate storage for simulation results:

``` r
nsim  <- 1000
n     <- 16000
m     <- 1000
p     <- 0.05
kappa <- 1
fpr   <- kappa / sqrt(n)

b0    <- 10
b1    <- 1
a0    <- 0.3
a1    <- 0.5

#pre-allocated storage
B <- array(0, dim = c(nsim, 4, 2))
S <- array(0, dim = c(nsim, 4, 2))

update_results <- function(b, V, i, method_idx) {
  for (j in 1:2) {
    B[i, method_idx, j] <<- b[j]
    S[i, method_idx, j] <<- sqrt(max(V[j,j], 0))
  }
}
```

Function to generate data:

``` r
generate_data <- function(n, m, p, fpr, b0, b1, a0, a1) {
  N    <- n + m
  X    <- numeric(N)
  Xhat <- numeric(N)
  u    <- runif(N)
  
  for (j in seq_len(N)) {
    if      (u[j] <= fpr)           X[j]   <- 1
    else if (u[j] <= 2*fpr)         Xhat[j]<- 1
    else if (u[j] <= p + fpr) {     # true positive
      X[j]   <- 1
      Xhat[j]<- 1
    }
  }
  
  eps <- rnorm(N)  # N(0,1)
  # heteroskedastic noise: a1 when X=1, a0 when X=0
  Y <- b0 + b1*X + (a1*X + a0*(1 - X)) * eps
  
  train_Y   <- Y[1:n]
  train_X   <- cbind(Xhat[1:n],        rep(1, n))
  test_Xhat <- cbind(Xhat[(n+1):N],    rep(1, m))
  test_X    <- cbind(X[(n+1):N],       rep(1, m))
  
  list(
    train_Y   = train_Y,
    train_X   = train_X,
    test_Xhat = test_Xhat,
    test_X    = test_X
  )
}
```

Generate data, implement methods, and store results:

``` r
for (i in seq_len(nsim)) {
  dat       <- generate_data(n, m, p, fpr, b0, b1, a0, a1)
  train_Y   <- dat$train_Y    # response variable in main sample
  train_X   <- dat$train_X    # generated labels in main sample
  test_Xhat <- dat$test_Xhat  # generated labels in validation sample
  test_X    <- dat$test_X     # true labels in validation sample
  
  # Method 1: run OLS on generated labels in the main sample (biased)
  ols_res <- ols(train_Y, train_X)
  update_results(ols_res$coef, ols_res$vcov, i, 1)
  
  # Method 2: Additive bias correction
  fpr_hat <- mean(test_Xhat[,1] * (1 - test_X[,1]))
  bca_res <- ols_bca(train_Y, train_X, fpr_hat, m, intercept=FALSE)
  update_results(bca_res$coef, bca_res$vcov, i, 2)

  # Method 3: Multiplicative bias correction
  bcm_res <- ols_bcm(train_Y, train_X, fpr_hat, m, intercept=FALSE)
  update_results(bcm_res$coef, bcm_res$vcov, i, 3)
  
  # Method 4: One-step estimator
  one_res <- one_step(train_Y, train_X, intercept=FALSE)
  update_results(one_res$coef, one_res$cov, i, 4)
  
  if (i %% 100 == 0) {
    message("Completed ", i, " of ", nsim, " sims")
  }
}
```

    ## Completed 100 of 1000 sims

    ## Completed 200 of 1000 sims

    ## Completed 300 of 1000 sims

    ## Completed 400 of 1000 sims

    ## Completed 500 of 1000 sims

    ## Completed 600 of 1000 sims

    ## Completed 700 of 1000 sims

    ## Completed 800 of 1000 sims

    ## Completed 900 of 1000 sims

    ## Completed 1000 of 1000 sims

Compute coverage probabilities of 95% confidence intervals for the slope
coefficient across methods:

``` r
coverage <- function(bgrid, b, se) {
  n_grid <- length(bgrid)
  cvg    <- numeric(n_grid)
  for (i in seq_along(bgrid)) {
    val      <- bgrid[i]
    cvg[i]   <- mean(abs(b - val) <= 1.96 * se)
  }
  cvg
}

true_beta1 <- b1

methods <- c(
  "OLS     " = 1,
  "ols_bca " = 2,
  "ols_bcm " = 3,
  "one_step" = 4
)

cov_dict <- sapply(methods, function(col) {
  slopes <- B[, col, 1]   
  ses    <- S[, col, 1]
  mean(abs(slopes - true_beta1) <= 1.96 * ses)
})

cov_series <- setNames(cov_dict, names(methods))
print(cov_series)
```

    ## OLS      ols_bca  ols_bcm  one_step 
    ##    0.000    0.835    0.879    0.950

Evidently, standard OLS confidence intervals for the slope coefficient
have coverage of zero. Both `ols_bca` and `ols_bcm` yield confidence
intervals with coverage probabilities a bit below the nominal level of
95%, but their coverage approaches 95% in larger sample sizes. Moreover,
`one_step` produces confidence intervals with coverage close to 95%.

Finally, we tabulate results, presenting:

-   the average estimate and average standard error across simulations
    for each method
-   intervals containing the 2.5% and 97.5% quantiles of the estimates
    across simultaions for each method

``` r
method_names <- names(methods)

coef_names <- c("Beta1","Beta0")

nmethods <- dim(B)[2]
df <- data.frame(Method = method_names, stringsAsFactors = FALSE)

df$Avg_Beta1        <- NA_real_
df$Avg_SE_Beta1     <- NA_real_
df$Quantiles_Beta1  <- NA_character_
df$Avg_Beta0        <- NA_real_
df$Avg_SE_Beta0     <- NA_real_
df$Quantiles_Beta0  <- NA_character_

for(i in seq_len(nmethods)) {
  est1 <- B[, i, 1]
  est0 <- B[, i, 2]
  se1  <- S[, i, 1]
  se0  <- S[, i, 2]
  
  ci1 <- quantile(est1, probs = c(0.025, 0.975))
  ci0 <- quantile(est0, probs = c(0.025, 0.975))
  
  df$Avg_Beta1[i]  <- mean(est1)
  df$Avg_SE_Beta1[i]   <- mean(se1)
  df$Quantiles_Beta1[i] <- sprintf("[%0.3f, %0.3f]", ci1[1], ci1[2])
  
  df$Avg_Beta0[i]  <- mean(est0)
  df$Avg_SE_Beta0[i]   <- mean(se0)
  df$Quantiles_Beta0[i] <- sprintf("[%0.3f, %0.3f]", ci0[1], ci0[2])
}

print(df)
```

    ##     Method Avg_Beta1 Avg_SE_Beta1 Quantiles_Beta1 Avg_Beta0 Avg_SE_Beta0
    ## 1 OLS      0.8335635   0.02128098  [0.790, 0.872]  10.00840  0.002559407
    ## 2 ols_bca  0.9695534   0.05143863  [0.871, 1.086]  10.00160  0.003509529
    ## 3 ols_bcm  1.0011194   0.06376873  [0.881, 1.181]  10.00002  0.003929130
    ## 4 one_step 0.9989799   0.03106329  [0.934, 1.056]  10.00010  0.002499879
    ##    Quantiles_Beta0
    ## 1 [10.003, 10.013]
    ## 2  [9.994, 10.008]
    ## 3  [9.990, 10.007]
    ## 4  [9.995, 10.005]

We see that OLS estimator of the slope coefficient is biased (it
under-estimates the true effect size by about 17% on average), while
`ols_bca`, `ols_bcm`, and `one_step` yield estimates close to the true
value of the slope coefficient.
