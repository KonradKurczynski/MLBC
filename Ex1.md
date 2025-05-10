``` r
library(MLBC)
```

# Example 1

## Parameters

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

alpha <- c(0.0, 0.5, 0.5)
beta  <- c(0.0, 2.0, 4.0)

#pre-allocated storage
B <- array(0, dim = c(nsim, 9, 2))
S <- array(0, dim = c(nsim, 9, 2))

update_results <- function(b, V, i, method_idx) {
  for (j in 1:2) {
    B[i, method_idx, j] <<- b[j]
    S[i, method_idx, j] <<- sqrt(max(V[j,j], 0))
  }
}
```

## Data Generation Process

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
  # heteroskedastic noise: σ₁ when X=1, σ₀ when X=0
  Y <- b0 + b1*X + (a1*X + a0*(1 - X)) * eps
  
  # split into train vs test
  train_Y   <- Y[       1:n    ]
  train_X   <- cbind(Xhat[    1:n],    rep(1, n))
  test_Y    <- Y[(n+1):N      ]
  test_Xhat <- cbind(Xhat[(n+1):N],    rep(1, m))
  test_X    <- cbind(X[(n+1):N],       rep(1, m))
  
  list(
    train_Y   = train_Y,
    train_X   = train_X,
    test_Y    = test_Y,
    test_Xhat = test_Xhat,
    test_X    = test_X
  )
}
```

## Estimation and bias correction

``` r
for (i in seq_len(nsim)) {
  dat   <- generate_data(n, m, p, fpr, b0, b1, a0, a1)
  tY    <- dat$train_Y;    tX <- dat$train_X
  eY    <- dat$test_Y;     eXhat <- dat$test_Xhat;  eX <- dat$test_X
  
  # 1) OLS on unlabeled (X̂)
  ols_res <- ols(tY, tX)
  update_results(ols_res$coef,  ols_res$vcov, i, 1)
  
  # 2) OLS on labeled (true X)
  ols_res <- ols(eY, eX)
  update_results(ols_res$coef,  ols_res$vcov, i, 2)
  
  # 3–8) Additive & multiplicative bias‑corrections
  fpr_hat <- mean(eXhat[,1] * (1 - eX[,1]))
  for (j in 1:3) {
    fpr_bayes <- (fpr_hat * m + alpha[j]) / (m + alpha[j] + beta[j])
    
    bca_res <- ols_bca(tY, tX, fpr_bayes, m)
    update_results(bca_res$coef, bca_res$vcov, i, 2 + j)
    
    bcm_res <- ols_bcm(tY, tX, fpr_bayes, m)
    update_results(bcm_res$coef, bcm_res$vcov, i, 5 + j)
  }
  
  # 9) One‑step estimator
  one_res <- one_step(tY, tX)
  update_results(one_res$coef, one_res$cov, i, 9)
  
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

true_beta1 <- 1.0

methods <- c(
  "OLS ĥ"  = 1,
  "OLS θ̂"  = 2,
  "BCA-0"  = 3,
  "BCA-1"  = 4,
  "BCA-2"  = 5,
  "BCM-0"  = 6,
  "BCM-1"  = 7,
  "BCM-2"  = 8,
  "OSU"    = 9
)

cov_dict <- sapply(methods, function(col) {
  slopes <- B[, col, 1]   
  ses    <- S[, col, 1]
  mean(abs(slopes - true_beta1) <= 1.96 * ses)
})

cov_series <- setNames(cov_dict, names(methods))
print(cov_series)
```

    ## OLS ĥ OLS θ̂ BCA-0 BCA-1 BCA-2 BCM-0 BCM-1 BCM-2   OSU 
    ## 0.000 0.952 0.841 0.890 0.888 0.887 0.912 0.911 0.951

``` r
method_names <- names(methods)

coef_names <- c("Beta1","Beta0")

nmethods <- dim(B)[2]
df <- data.frame(Method = method_names, stringsAsFactors = FALSE)

df$Est_Beta1   <- NA_real_
df$SE_Beta1    <- NA_real_
df$CI95_Beta1  <- NA_character_
df$Est_Beta0   <- NA_real_
df$SE_Beta0    <- NA_real_
df$CI95_Beta0  <- NA_character_

for(i in seq_len(nmethods)) {
  est1 <- B[, i, 1]; se1 <- S[, i, 1]
  est0 <- B[, i, 2]; se0 <- S[, i, 2]
  
  ci1 <- quantile(est1, probs = c(0.025, 0.975))
  ci0 <- quantile(est0, probs = c(0.025, 0.975))
  
  df$Est_Beta1[i]  <- mean(est1)
  df$SE_Beta1[i]   <- mean(se1)
  df$CI95_Beta1[i] <- sprintf("[%0.3f, %0.3f]", ci1[1], ci1[2])
  
  df$Est_Beta0[i]  <- mean(est0)
  df$SE_Beta0[i]   <- mean(se0)
  df$CI95_Beta0[i] <- sprintf("[%0.3f, %0.3f]", ci0[1], ci0[2])
}

print(df)
```

    ##   Method Est_Beta1   SE_Beta1     CI95_Beta1 Est_Beta0    SE_Beta0
    ## 1  OLS ĥ 0.8345230 0.02126622 [0.793, 0.876] 10.008229 0.002559755
    ## 2  OLS θ̂ 0.9979584 0.07110668 [0.862, 1.139]  9.999644 0.009712562
    ## 3  BCA-0 0.9733887 0.05192212 [0.879, 1.097] 10.001295 0.003527102
    ## 4  BCA-1 0.9818128 0.05328685 [0.888, 1.105] 10.000874 0.003578399
    ## 5  BCA-2 0.9815196 0.05324235 [0.888, 1.105] 10.000889 0.003576679
    ## 6  BCM-0 1.0064253 0.06465110 [0.886, 1.197]  9.999649 0.003963054
    ## 7  BCM-1 1.0188792 0.06713945 [0.896, 1.213]  9.999027 0.004060987
    ## 8  BCM-2 1.0184172 0.06705113 [0.896, 1.212]  9.999050 0.004057390
    ## 9    OSU 0.9985015 0.03102350 [0.934, 1.055]  9.999928 0.002500163
    ##         CI95_Beta0
    ## 1 [10.003, 10.013]
    ## 2  [9.981, 10.018]
    ## 3  [9.994, 10.007]
    ## 4  [9.994, 10.007]
    ## 5  [9.994, 10.007]
    ## 6  [9.990, 10.007]
    ## 7  [9.989, 10.007]
    ## 8  [9.989, 10.007]
    ## 9  [9.995, 10.005]
