# CEO Time Use and Firm Performance: A Topic Model Application
``` r
library(MLBC)
```

# About this notebook

This notebook estimates the association between CEO time alocation and
firm performance [(Bandiera et
al. 2020)](https://www.journals.uchicago.edu/doi/10.1086/705331). It
illustrates how the functions `ols_bca_topic` and `ols_bcm_topic` can be
used to correct bias from estimated topic model shares. The notebook
reproduces results from Table 2 of [Battaglia, Christensen, Hansen &
Sacher (2024)](https://arxiv.org/abs/2402.15585).

[(Bandiera et
al. 2020)](https://www.journals.uchicago.edu/doi/10.1086/705331) conduct
a time-use survey for a sample of CEOs. Survey responses are recorded
for each 15-minute interval of a given week. The sample consists of 654
answer combinations. To reduce dimensionality, the authors fit a topic
model with two topics. One topic places relatively higher mass on
features associated with “management,” like visiting production sites or
meeting with suppliers, while the other places relatively higher mass on
features associated with “leadership” like communicating with other
C-suite executives and holding large, multi-function meetings.

Each CEO’s leadership weight is a measure of their tendency to engage in
leadership activities. One of the key results in [(Bandiera et
al. 2020)](https://www.journals.uchicago.edu/doi/10.1086/705331) is a
regression of log sales, a measure of firm size, on the estimated
leadership weight (along with other firm controls).

## Data

The package contains `topic_model_data`, which we will use for the
regression as well as joint estimates of the regression and topic model,
as described in [Battaglia, Christensen, Hansen & Sacher
(2024)](https://arxiv.org/abs/2402.15585).

``` r
# Load the topic model data
data("topic_model_data")


# Extract components
Z <- as.matrix(topic_model_data$covars)              # Control variables
estimation_data <- topic_model_data$estimation_data   # Main dataset  
gamma_draws <- as.matrix(topic_model_data$gamma_draws) # MCMC draws
theta_est_full <- as.matrix(topic_model_data$theta_est_full)   # Full sample topic estimates
theta_est_samp <- as.matrix(topic_model_data$theta_est_samp)   # Subsample topic estimates
beta_est_full <- as.matrix(topic_model_data$beta_est_full)     # Full sample topic-word distributions
beta_est_samp <- as.matrix(topic_model_data$beta_est_samp)     # Subsample topic-word distributions
lda_data <- as.matrix(topic_model_data$lda_data)      # LDA validation data

# Dependent variable: log employment, country fixed effects, and survey-wave fixed effects
Y <- estimation_data$ly
sigma_y <- sd(Y)

# Show sample of the data
sample_data <- data.frame(
  Y = Y,
  theta_topic1 = theta_est_full[, 1], 
  control1 = Z[, 1], 
  control2 = Z[, 2]
)
head(sample_data)
#>          Y theta_topic1  control1 control2
#> 1 12.35214   0.60518616 1.2681264        0
#> 2 10.09636   0.08448929 1.1132969        0
#> 3 14.07556   0.96903920 3.2279463        0
#> 4 12.35838   0.28817785 1.6723140        0
#> 5 10.53030   0.43081099 0.7284679        0
#> 6 11.95783   0.34403238 1.5895084        0
```

Here, `theta_topic1` contains the leadership topic weight for each
observation.

# Results

We first present results for an OLS regression of log sales on the
leadership topic weight and controls:

``` r
# Full sample two-step estimation
theta_full <- theta_est_full
Xhat_full <- cbind(theta_full[, 1], Z)  # First topic + controls
lm_full <- ols(Y, Xhat_full, se = TRUE, intercept = TRUE)
summary(lm_full)
#> 
#> MLBC Model Summary
#> ==================
#> 
#> Formula:  Y ~ Beta_0 + Beta_1 * X1 + Beta_2 * X2 + Beta_3 * X3 + Beta_4 * X4 + Beta_5 * X5 + Beta_6 * X6 + Beta_7 * X7 + Beta_8 * X8 + Beta_9 * X9 + Beta_10 * X10 + Beta_11 * X11 + Beta_12 * X12 
#> 
#> 
#> Coefficients:
#> 
#>         Estimate Std.Error z.value Pr(>|z|) Signif             95% CI
#> Beta_0    9.8741    0.1592 62.0256  < 2e-16    ***  [9.5621, 10.1861]
#> Beta_1    0.4047    0.0921  4.3946 1.11e-05    ***   [0.2242, 0.5851]
#> Beta_2    1.2111    0.0286 42.3144  < 2e-16    ***   [1.1550, 1.2672]
#> Beta_3   -0.0428    0.3921 -0.1092   0.9131         [-0.8112, 0.7256]
#> Beta_4    0.2767    0.1983  1.3956   0.1628         [-0.1119, 0.6654]
#> Beta_5    0.5602    0.1920  2.9183   0.0035     **   [0.1840, 0.9364]
#> Beta_6    0.5918    0.1761  3.3604   0.0008    ***   [0.2466, 0.9370]
#> Beta_7    0.3369    0.1632  2.0650   0.0389      *   [0.0171, 0.6567]
#> Beta_8    0.7628    0.0900  8.4721  < 2e-16    ***   [0.5863, 0.9393]
#> Beta_9    0.8282    0.0790 10.4884  < 2e-16    ***   [0.6734, 0.9830]
#> Beta_10   0.6806    0.0820  8.2967  < 2e-16    ***   [0.5198, 0.8414]
#> Beta_11  -0.3406    0.1261 -2.7015   0.0069     ** [-0.5876, -0.0935]
#> Beta_12  -0.0433    0.1045 -0.4148   0.6783         [-0.2482, 0.1615]
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

We now compare these estimates with bias-corrected estimates. We will
use `ols_bca_topic`. This requires an estimate of $`\kappa`$, which is
$`\sqrt{n}E[C_{i}^{-1}]`$, where $`C_i`$ is the number of feature counts
in unstructured document $`i`$. This is stored in the first column of
`lda_data`.

## Bias Correction

The empirical analogue of this expression is **θ̂ᵢ** for the simple
model. While not directly applicable here as the topic model structure
is more complex, it still allows one to qualitatively compare sampling
error (reflected by **ζ̂ⱼ⁻¹**) and measurement error (reflected by **ε̂
\[ζ̂ⱼ⁻¹\]**). The empirical analogue of this expression is 0.44.

``` r
# Full sample bias correction
kappa_full <- mean(1.0 / lda_data[, 1]) * sqrt(nrow(lda_data))
print(c("Kappa: ", kappa_full))
#> [1] "Kappa: "           "0.441711240848792"
```

In addition to $`\kappa`$, we need to construct a matrix `S` which picks
off the relevant column of `theta_full` (a `n` by `K` matrix, `K` being
the number of topics, here `K = 2`) to include in the regression.

We also include the estmated topic-word distributions (a `V` by `K`
matrix, `V` being the number of features in the otpic model).

``` r
S <- matrix(c(1.0, 0.0), nrow = 1)  # Topic loadings: first topic active

bc_full <- ols_bca_topic(Y = Y,
                         Q = Z,                    # Control variables  
                         W = theta_est_full,       # Document-topic proportions
                         S = S,                    # Topic loadings
                         B = beta_est_full,        # Topic-word distributions
                         k = kappa_full)           # Scaling parameter
summary(bc_full)
#> 
#> MLBC Model Summary
#> ==================
#> 
#> Formula:  Y ~ Beta_0 + Beta_1 * topic1 + Beta_2 * V2 + Beta_3 * V3 + Beta_4 * V4 + Beta_5 * V5 + Beta_6 * V6 + Beta_7 * V7 + Beta_8 * V8 + Beta_9 * V9 + Beta_10 * V10 + Beta_11 * V11 + Beta_12 * V12 
#> 
#> Number of observations: 916 
#> 
#> Coefficients:
#> 
#>         Estimate Std.Error z.value Pr(>|z|) Signif             95% CI
#> Beta_0    9.8425    0.1592 61.8269  < 2e-16    ***  [9.5305, 10.1545]
#> Beta_1    0.4743    0.0921  5.1504 2.60e-07    ***   [0.2938, 0.6547]
#> Beta_2    1.2049    0.0286 42.0975  < 2e-16    ***   [1.1488, 1.2610]
#> Beta_3   -0.0322    0.3921 -0.0821   0.9346         [-0.8006, 0.7362]
#> Beta_4    0.2846    0.1983  1.4350   0.1513         [-0.1041, 0.6732]
#> Beta_5    0.5694    0.1920  2.9665   0.0030     **   [0.1932, 0.9457]
#> Beta_6    0.5982    0.1761  3.3969   0.0007    ***   [0.2531, 0.9434]
#> Beta_7    0.3419    0.1632  2.0956   0.0361      *   [0.0221, 0.6617]
#> Beta_8    0.7509    0.0900  8.3392  < 2e-16    ***   [0.5744, 0.9273]
#> Beta_9    0.8139    0.0790 10.3071  < 2e-16    ***   [0.6591, 0.9687]
#> Beta_10   0.6750    0.0820  8.2279  < 2e-16    ***   [0.5142, 0.8357]
#> Beta_11  -0.3389    0.1261 -2.6883   0.0072     ** [-0.5860, -0.0918]
#> Beta_12  -0.0482    0.1045 -0.4616   0.6444         [-0.2531, 0.1566]
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

The two methods (`ols` and `ols_bca_topic`) produce similar estimates
and confidence intervals. This suggests that measurement error in the
estimated topic_1 shares is small enough that it doesn’t materially
distort inference.

To explore this further, we repeat the above taking a 10% subsample of
the data used to estimate the topic model. This ensures that the
estimated topic weights are noisier signals of the true leadership
index. Here we are running the same regression as before, just with a
noisier vvalue of the topic_1 weight.

The data are names as before, with a `_samp` suffix.

``` r
# 10% Subsample two-step estimation  
theta_samp <- theta_est_samp
Xhat_samp <- cbind(theta_samp[, 1], Z)
lm_samp <- ols(Y, Xhat_samp, se = TRUE, intercept = TRUE)
summary(lm_samp)
#> 
#> MLBC Model Summary
#> ==================
#> 
#> Formula:  Y ~ Beta_0 + Beta_1 * X1 + Beta_2 * X2 + Beta_3 * X3 + Beta_4 * X4 + Beta_5 * X5 + Beta_6 * X6 + Beta_7 * X7 + Beta_8 * X8 + Beta_9 * X9 + Beta_10 * X10 + Beta_11 * X11 + Beta_12 * X12 
#> 
#> 
#> Coefficients:
#> 
#>         Estimate Std.Error z.value Pr(>|z|) Signif             95% CI
#> Beta_0    9.9405    0.1708 58.2022  < 2e-16    ***  [9.6058, 10.2753]
#> Beta_1    0.2267    0.1351  1.6779   0.0934      .  [-0.0381, 0.4915]
#> Beta_2    1.2372    0.0283 43.6941  < 2e-16    ***   [1.1817, 1.2927]
#> Beta_3   -0.0628    0.3977 -0.1580   0.8745         [-0.8424, 0.7167]
#> Beta_4    0.2608    0.1998  1.3053   0.1918         [-0.1308, 0.6523]
#> Beta_5    0.5308    0.1938  2.7386   0.0062     **   [0.1509, 0.9106]
#> Beta_6    0.5718    0.1774  3.2239   0.0013     **   [0.2242, 0.9194]
#> Beta_7    0.3223    0.1632  1.9745   0.0483      *   [0.0024, 0.6423]
#> Beta_8    0.8083    0.0917  8.8180  < 2e-16    ***   [0.6286, 0.9879]
#> Beta_9    0.8902    0.0792 11.2361  < 2e-16    ***   [0.7350, 1.0455]
#> Beta_10   0.6975    0.0834  8.3602  < 2e-16    ***   [0.5340, 0.8610]
#> Beta_11  -0.3626    0.1288 -2.8156   0.0049     ** [-0.6151, -0.1102]
#> Beta_12  -0.0233    0.1064 -0.2188   0.8268         [-0.2318, 0.1853]
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Evidently, increasing the measurement error in the estimated topic
weights reduces the estimated slope coefficient by around 50%. Moreover,
OLS confidence intervals for the slope coefficient now include zero.

We now compare with bias correction:

``` r
# 10% Subsample bias correction
kappa_samp <- mean(1.0 / lda_data[, 2]) * sqrt(nrow(lda_data))

bc_samp <- ols_bca_topic(Y = Y,
                         Q = Z,
                         W = theta_est_samp,
                         S = S, 
                         B = beta_est_samp,
                         k = kappa_samp)
summary(bc_samp)
#> 
#> MLBC Model Summary
#> ==================
#> 
#> Formula:  Y ~ Beta_0 + Beta_1 * topic1 + Beta_2 * V2 + Beta_3 * V3 + Beta_4 * V4 + Beta_5 * V5 + Beta_6 * V6 + Beta_7 * V7 + Beta_8 * V8 + Beta_9 * V9 + Beta_10 * V10 + Beta_11 * V11 + Beta_12 * V12 
#> 
#> Number of observations: 916 
#> 
#> Coefficients:
#> 
#>         Estimate Std.Error z.value Pr(>|z|) Signif             95% CI
#> Beta_0    9.5115    0.1708 55.6905  < 2e-16    ***   [9.1768, 9.8463]
#> Beta_1    1.0538    0.1351  7.7989 6.25e-15    ***   [0.7889, 1.3186]
#> Beta_2    1.2009    0.0283 42.4097  < 2e-16    ***   [1.1454, 1.2564]
#> Beta_3    0.0892    0.3977  0.2242   0.8226         [-0.6904, 0.8687]
#> Beta_4    0.3682    0.1998  1.8433   0.0653      .  [-0.0233, 0.7598]
#> Beta_5    0.6194    0.1938  3.1959   0.0014     **   [0.2395, 0.9993]
#> Beta_6    0.6351    0.1774  3.5809   0.0003    ***   [0.2875, 0.9827]
#> Beta_7    0.3749    0.1632  2.2968   0.0216      *   [0.0550, 0.6949]
#> Beta_8    0.7203    0.0917  7.8584 3.89e-15    ***   [0.5407, 0.9000]
#> Beta_9    0.8129    0.0792 10.2601  < 2e-16    ***   [0.6576, 0.9682]
#> Beta_10   0.6392    0.0834  7.6623 1.83e-14    ***   [0.4757, 0.8028]
#> Beta_11  -0.4079    0.1288 -3.1673   0.0015     ** [-0.6604, -0.1555]
#> Beta_12  -0.0538    0.1064 -0.5056   0.6131         [-0.2623, 0.1547]
#> ---
#> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Performing the bias correction results in a much larger estimated effect
size.

Finally, we tabulate the results from the joint estimation performed by
[Battaglia, Christensen, Hansen & Sacher
(2024)](https://arxiv.org/abs/2402.15585):

``` r
# Joint estimation using MCMC draws (scaled by dependent variable standard deviation)
gamma_scaled <- gamma_draws * sigma_y
gamma_hat_1 <- colMeans(gamma_scaled)

# Calculate empirical confidence intervals from MCMC draws
alpha <- 0.05
ci_lower_1 <- apply(gamma_scaled, 2, quantile, probs = alpha/2)
ci_upper_1 <- apply(gamma_scaled, 2, quantile, probs = 1 - alpha/2)

cat("Joint estimates and confidence intervals:\n")
#> Joint estimates and confidence intervals:
cat("Full Sample:", round(gamma_hat_1[1], 3), 
    " [", round(ci_lower_1[1], 3), ",", round(ci_upper_1[1], 3), "]\n")
#> Full Sample: 0.402  [ 0.24 , 0.603 ]
cat("10% Subsample:", round(gamma_hat_1[2], 3),
    " [", round(ci_lower_1[2], 3), ",", round(ci_upper_1[2], 3), "]\n")
#> 10% Subsample: 0.439  [ 0.153 , 0.711 ]
```

We see that unlike OLS estimation, joint estimation is robust to
increasing the noise in the estimated topic weight. Both samples produce
a similar estimated effect size onfidence intervals that exclude zero.

Finally, we tabulate all results together:

``` r
# Extract key coefficients for the first topic (management)
results <- data.frame(
  Sample = c("Full", "10% Subsample", "Full", "10% Subsample", "Full"),
  Method = c("Two-Step", "Two-Step", "Bias Correction", "Bias Correction", "Joint"),
  Estimate = c(lm_full$coef[1],
               lm_samp$coef[1], 
               bc_full$coef[1],
               bc_samp$coef[1],
               gamma_hat_1[1]),
  CI_Lower = c(lm_full$coef[1] - 1.96 * sqrt(lm_full$vcov[1,1]),
               lm_samp$coef[1] - 1.96 * sqrt(lm_samp$vcov[1,1]),
               bc_full$coef[1] - 1.96 * sqrt(bc_full$vcov[1,1]),
               bc_samp$coef[1] - 1.96 * sqrt(bc_samp$vcov[1,1]),
               ci_lower_1[1]),
  CI_Upper = c(lm_full$coef[1] + 1.96 * sqrt(lm_full$vcov[1,1]),
               lm_samp$coef[1] + 1.96 * sqrt(lm_samp$vcov[1,1]),
               bc_full$coef[1] + 1.96 * sqrt(bc_full$vcov[1,1]),
               bc_samp$coef[1] + 1.96 * sqrt(bc_samp$vcov[1,1]),
               ci_upper_1[1])
)

results$Estimate <- round(results$Estimate, 3)
results$CI_Lower <- round(results$CI_Lower, 3)
results$CI_Upper <- round(results$CI_Upper, 3)

print(results)
#>          Sample          Method Estimate CI_Lower CI_Upper
#> 1          Full        Two-Step    0.405    0.224    0.585
#> 2 10% Subsample        Two-Step    0.227   -0.038    0.492
#> 3          Full Bias Correction    0.474    0.294    0.655
#> 4 10% Subsample Bias Correction    1.054    0.789    1.319
#> 5          Full           Joint    0.402    0.240    0.603
```

``` r
if (require(ggplot2, quietly = TRUE)) {
  plot_data <- results
  plot_data$Method_Sample <- paste(plot_data$Method, "(", plot_data$Sample, ")")
  plot_data$Type <- ifelse(plot_data$Method == "Two-Step", "Two-Step",
                   ifelse(plot_data$Method == "Bias Correction", "Bias Correction", "Joint"))
  
  p <- ggplot(plot_data, aes(x = reorder(Method_Sample, Estimate), y = Estimate, color = Type)) +
    geom_point(size = 3) +
    geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), width = 0.3, size = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
    coord_flip() +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      panel.grid.minor = element_blank(),
      plot.title = element_text(size = 12, face = "bold"),
      plot.subtitle = element_text(size = 10)
    ) +
    labs(
      title = "Estimates of Impact of CEO Behavior on Firm Performance",
      subtitle = "Comparison of estimation strategies with 95% confidence intervals",
      x = "Estimation Strategy",
      y = "Coefficient Estimate",
      color = "Method Type"
    ) +
    scale_color_manual(values = c("Two-Step" = "#56B4E9", 
                                 "Bias Correction" = "#009E73",
                                 "Joint" = "#E69F00"))
  
  print(p)
} else {
  cat("ggplot2 not available for plotting\n")
}
```

![](topic_model_example_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->
