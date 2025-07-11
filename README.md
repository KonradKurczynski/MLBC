# MLBC
`MLBC` is an R package for correcting bias and performing valid inference in regressions that include variables generated by AI/ML methods. The bias-correction methods are described in [Battaglia, Christensen, Hansen & Sacher (2024)](https://arxiv.org/abs/2402.15585). 

## Requirements and installation

`MLBC` runs on R 3.5 or above and uses `TMB`. It can be installed from CRAN by running

To install the package, run 
```
pip install ValidMLInference
```
in your R console. 

## Using ValidMLInference

To get started, we recommend looking at the following examples and resources: 
1. [**Remote Work**](https://github.com/KonradKurczynski/MLBC/blob/main/remote_work.md): This notebook estimates the association between working from home and salaries using real-world job postings data [(Hansen et al., 2023)](https://dx.doi.org/10.2139/ssrn.4380734). It illustrates how the functions `ols_bca`, `ols_bcm` and `one_step` can be used to correct bias from regressing on AI/ML-generated labels. The notebook reproduces results from Table 1 of [Battaglia, Christensen, Hansen & Sacher (2024)](https://arxiv.org/abs/2402.15585).
2. [**Topic Models**](https://github.com/KonradKurczynski/MLBC/blob/main/topic_model_example.md): This notebook estimates the association between CEO time allocation and firm performance [(Bandiera et al. 2020)](https://doi.org/10.1086/705331). It illustrates how the functions `ols_bca_topic` and `ols_bcm_topic` can be used to correct bias from estimated topic model shares. The notebook reproduces results from Table 2 of [Battaglia, Christensen, Hansen & Sacher (2024)](https://arxiv.org/abs/2402.15585).
3. [**Synthetic Example**](https://github.com/KonradKurczynski/MLBC/blob/main/synthetic_example.md): A synthetic example comparing the performance of different bias-correction methods in the context of AI/ML-generated labels.
4. [**Manual**](https://github.com/KonradKurczynski/MLBC/blob/main/MLBC-manual.pdf): A detailed reference describing all available functions, optional arguments, and usage tips.

## Quickstart 
Code below compares coefficients obtained by ordinary least squares methods and those obtained by the `one_step` approach, when used on variables subject to classification error. We can see that the 95% confidence interval generated by `one_step` contains the true parameter of 2, whereas the standard ols approach doesn't.

``` r
library(MLBC)

# Generate synthetic data with mislabeling
n <- 1000
true_effect <- 2.0

# True treatment assignment
X_true <- rbinom(n, 1, 0.5)

# Observed (mislabeled) treatment with 20% error rate
mislabel_prob <- 0.2
X_obs <- X_true
mislabel_mask <- rbinom(n, 1, mislabel_prob) == 1
X_obs[mislabel_mask] <- 1 - X_obs[mislabel_mask]

# Generate outcome with true treatment effect
Y <- 1.0 + true_effect * X_true + rnorm(n, 0, 1)

# Create DataFrame
data <- data.frame(Y = Y, X_obs = X_obs)

# Naive OLS using mislabeled data
ols_result <- ols(Y ~ X_obs, data = data)
print("OLS Results (using mislabeled data):")
#> [1] "OLS Results (using mislabeled data):"
print(summary(ols_result))
#> 
#> MLBC Model Summary
#> ==================
#> 
#> Formula:  Y ~ Beta_0 + Beta_1 * X_obs 
#> 
#> 
#> Coefficients:
#> 
#>        Estimate Std.Error z.value Pr(>|z|) Signif           95% CI
#> Beta_0   1.3346    0.0568 23.4937  < 2e-16    *** [1.2233, 1.4459]
#> Beta_1   1.2471    0.0809 15.4229  < 2e-16    *** [1.0886, 1.4056]
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# One-step estimator that corrects for mislabeling
one_step_result <- one_step(Y ~ X_obs, data = data)
print("\nOne-Step Results (correcting for mislabeling):")
#> [1] "\nOne-Step Results (correcting for mislabeling):"
print(summary(one_step_result))
#> 
#> MLBC Model Summary
#> ==================
#> 
#> Formula:  Y ~ Beta_0 + Beta_1 * X_obs 
#> 
#> Number of observations: 1000 
#> Log-likelihood: -2344.289 
#> 
#> Coefficients:
#> 
#>        Estimate Std.Error z.value Pr(>|z|) Signif           95% CI
#> Beta_0   0.9443    0.0852 11.0868  < 2e-16    *** [0.7774, 1.1113]
#> Beta_1   1.9803    0.1009 19.6202  < 2e-16    *** [1.7825, 2.1781]
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

# Extract confidence intervals
ols_ci <- confint(ols_result)["X_obs", ]
one_step_ci <- confint(one_step_result)["X_obs", ]

cat("\nTrue treatment effect:", true_effect, "\n")
#> 
#> True treatment effect: 2
cat("OLS 95% CI contains true value:", 
    ols_ci[1] <= true_effect && true_effect <= ols_ci[2], "\n")
#> OLS 95% CI contains true value: FALSE
cat("One-step 95% CI contains true value:", 
    one_step_ci[1] <= true_effect && true_effect <= one_step_ci[2], "\n")
#> One-step 95% CI contains true value: TRUE
```
