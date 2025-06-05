# Simulation Example: Bias Correction in Regression with Generated Binary Labels
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

# Pre-allocated storage
B <- array(0, dim = c(nsim, 4, 2))
S <- array(0, dim = c(nsim, 4, 2))

update_results <- function(fit_obj, i, method_idx) {
  coefs <- coef(fit_obj)
  vcov_mat <- vcov(fit_obj)
  ses <- sqrt(pmax(diag(vcov_mat), 0))
  
  B[i, method_idx, ] <<- coefs
  S[i, method_idx, ] <<- ses
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
    else if (u[j] <= 2 * fpr)       Xhat[j]<- 1
    else if (u[j] <= p + fpr) {     # true positive
      X[j]   <- 1
      Xhat[j]<- 1
    }
    # otherwise X[j] and Xhat[j] stay 0
  }
  
  eps <- rnorm(N)   # N(0,1)
  # heteroskedastic noise: σ₁ when X=1, σ₀ when X=0
  Y <- b0 + b1 * X + (a1 * X + a0 * (1 - X)) * eps
  
  # Create data frames for train and test
  train_df <- data.frame(
    Y    = Y[1:n],
    Xhat = Xhat[1:n]
  )
  
  test_df <- data.frame(
    Y    = Y[(n+1):N],
    Xhat = Xhat[(n+1):N],
    X    = X[(n+1):N]
  )
  
  list(
    train_df = train_df,
    test_df  = test_df
  )
}
```

Generate data, implement methods, and store results:

``` r
for (i in seq_len(nsim)) {
  dat <- generate_data(n, m, p, fpr, b0, b1, a0, a1)
  
  # Method 1: OLS on generated labels (biased)
  fit_ols <- ols(Y ~ Xhat, data = dat$train_df)
  update_results(fit_ols, i, 1)
  
  # Estimate false positive rate from validation sample
  fpr_hat <- mean((dat$test_df$Xhat == 1) & (dat$test_df$X == 0))
  
  # Method 2: Additive bias correction
  fit_bca <- ols_bca(Y ~ Xhat, data = dat$train_df, fpr = fpr_hat, m = m)
  update_results(fit_bca, i, 2)
  
  # Method 3: Multiplicative bias correction
  fit_bcm <- ols_bcm(Y ~ Xhat, data = dat$train_df, fpr = fpr_hat, m = m)
  update_results(fit_bcm, i, 3)
  
  # Method 4: One-step estimator
  fit_onestep <- one_step(Y ~ Xhat, data = dat$train_df, 
                          homoskedastic = FALSE, distribution = "normal")
  update_results(fit_onestep, i, 4)
  
  if (i %% 100 == 0) {
    message("Completed ", i, " of ", nsim, " simulations")
  }
}
```

    ## Completed 100 of 1000 simulations

    ## Completed 200 of 1000 simulations

    ## Completed 300 of 1000 simulations

    ## Completed 400 of 1000 simulations

    ## Completed 500 of 1000 simulations

    ## Completed 600 of 1000 simulations

    ## Completed 700 of 1000 simulations

    ## Completed 800 of 1000 simulations

    ## Completed 900 of 1000 simulations

    ## Completed 1000 of 1000 simulations

Compute coverage probabilities of 95% confidence intervals for the slope
coefficient across methods:

``` r
true_beta1 <- b1

methods <- c(
  "OLS      " = 1,
  "ols_bca  " = 2,
  "ols_bcm  " = 3,
  "one_step " = 4
)

coverage_probs <- sapply(methods, function(col) {
  slopes <- B[, col, 1]   # slope coefficients (first parameter)
  ses    <- S[, col, 1]   # standard errors for slopes
  mean(abs(slopes - true_beta1) <= 1.96 * ses)
})

names(coverage_probs) <- names(methods)
print(coverage_probs)
```

    ## OLS       ols_bca   ols_bcm   one_step  
    ##     0.000     0.850     0.894     0.945

Evidently, standard OLS confidence intervals for the slope coefficient
have coverage of zero. Both `ols_bca` and `ols_bcm` yield confidence
intervals with coverage probabilities a bit below the nominal level of
95%, but their coverage approaches 95% in larger sample sizes. Moreover,
`one_step` produces confidence intervals with coverage close to 95%.

Finally, we tabulate results, presenting:

- the average estimate and average standard error across simulations for
  each method;
- intervals containing the 2.5% and 97.5% quantiles of the estimates
  across simultaions for each method.

``` r
method_names <- names(methods)
nmethods <- length(methods)

results_df <- data.frame(
  Method = method_names,
  stringsAsFactors = FALSE
)

# Initialize columns
results_df$Avg_Beta1       <- NA_real_
results_df$Avg_SE_Beta1    <- NA_real_
results_df$Quantiles_Beta1 <- NA_character_
results_df$Avg_Beta0       <- NA_real_
results_df$Avg_SE_Beta0    <- NA_real_
results_df$Quantiles_Beta0 <- NA_character_

for (i in seq_len(nmethods)) {
  # Extract estimates for slope (Beta1) and intercept (Beta0)
  est_slope <- B[, i, 1]  # slope coefficients
  est_int   <- B[, i, 2]  # intercept coefficients
  se_slope  <- S[, i, 1]  # slope standard errors
  se_int    <- S[, i, 2]  # intercept standard errors
  
  # Compute quantiles
  ci_slope <- quantile(est_slope, probs = c(0.025, 0.975))
  ci_int   <- quantile(est_int, probs = c(0.025, 0.975))
  
  # Fill in results
  results_df$Avg_Beta1[i]       <- mean(est_slope)
  results_df$Avg_SE_Beta1[i]    <- mean(se_slope)
  results_df$Quantiles_Beta1[i] <- sprintf("[%0.3f, %0.3f]", ci_slope[1], ci_slope[2])
  
  results_df$Avg_Beta0[i]       <- mean(est_int)
  results_df$Avg_SE_Beta0[i]    <- mean(se_int)
  results_df$Quantiles_Beta0[i] <- sprintf("[%0.3f, %0.3f]", ci_int[1], ci_int[2])
}

print(results_df)
```

    ##      Method Avg_Beta1 Avg_SE_Beta1 Quantiles_Beta1 Avg_Beta0 Avg_SE_Beta0
    ## 1 OLS       0.8337209   0.02126083  [0.792, 0.875] 10.008334  0.002560337
    ## 2 ols_bca   0.9703655   0.05153794  [0.874, 1.085] 10.001505  0.003514317
    ## 3 ols_bcm   1.0021273   0.06393480  [0.877, 1.176]  9.999921  0.003936437
    ## 4 one_step  0.9978378   0.03104741  [0.930, 1.058] 10.000038  0.002500850
    ##    Quantiles_Beta0
    ## 1 [10.003, 10.013]
    ## 2  [9.994, 10.009]
    ## 3  [9.991, 10.008]
    ## 4  [9.995, 10.005]

We see that OLS estimator of the slope coefficient is biased (it
under-estimates the true effect size by about 17% on average), while
`ols_bca`, `ols_bcm`, and `one_step` yield estimates close to the true
value of the slope coefficient.
