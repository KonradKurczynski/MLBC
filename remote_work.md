# Remote Work and Wage Inequality: Correcting Bias in Regression with Generated Binary Labels
## About this notebook

This notebook estimates the association between working from home and
salaries using real-world job postings data [(Hansen et al.,
2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4380734). It
illustrates how the functions `ols_bca`, `ols_bcm`, and `one_step` can
be used to correct bias from regressing on AI/ML-generated labels. The
notebook reproduces results from Table 1 of [Battaglia, Christensen,
Hansen & Sacher (2024)](https://arxiv.org/abs/2402.15585).

``` r
library(MLBC)
```

## The Dataset

The package contains a subset of [a larger dataset](https://wfhmap.com)
regarding work from home. The sample consists of 16,315 job postings for
2022 and 2023 with “San Diego, CA” recorded as the city and “72”
recorded as the NAICS2 industry code of the advertising firm.

The dataset contains the following entries:

1.  `city_name`: city of the job posting

2.  `naics_2022_2` : type of a business

3.  `id`: unique identifier of the job posting

4.  `salary`: salary offered

5.  `wfh_wham`: **ML-generated indicator of whether the job ofers work
    from home using fine-tuned DistilBERT as in [(Hansen et al.,
    2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4380734)**

6.  `soc_2021_2`: Bureau of Labor Statistics Standard Occupational
    Classification code

7.  `employment_type_name`: indicates whether the position is full-tuime
    or part-time

``` r
SD_data <- MLBC::SD_data
SD_data$salary <- log(SD_data$salary)
head(SD_data)
```

    ##       city_name naics_2022_2                                       id   salary
    ## 1 San Diego, CA           72 002e22ebe1b837ac6b0cebcbb720613138765f51 10.95954
    ## 2 San Diego, CA           72 00442454060b60c1c0ad4ed78bc29111935f400b 10.34817
    ## 3 San Diego, CA           72 007a1c1a527ed15006705379cec780aaae4930af 10.41271
    ## 4 San Diego, CA           72 00991b69215b1cc14c08c4cdfa1b10bbbdf6ceba 10.61054
    ## 5 San Diego, CA           72 00edf6dc0abb731a0befa73f6748ff3f5ce842f4 10.73117
    ## 6 San Diego, CA           72 01b2f3e54547ccac8a7386458d06ee3f6fbf45ba 10.41271
    ##   wfh_wham soc_2021_2   employment_type_name
    ## 1        0    11-0000 Full-time (> 32 hours)
    ## 2        0    35-0000 Full-time (> 32 hours)
    ## 3        0    35-0000  Part-time / full-time
    ## 4        0    35-0000 Full-time (> 32 hours)
    ## 5        0    11-0000 Full-time (> 32 hours)
    ## 6        0    35-0000 Full-time (> 32 hours)

# Estimating the false-positive rate

The variable `wfh_wham` describing whether the job posting offers remote
work is not manually collected, but it is imputed via ML methods using
fine-tuned DistilBERT as in [(Hansen et al.,
2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4380734).
This classifier has over 99% test accuracy. Nevertheless as [Battaglia,
Christensen, Hansen & Sacher (2024)](https://arxiv.org/abs/2402.15585)
document, even high-performance classifiers can lead to large biases in
OLS estimates.

The bias correction methods `ols_bca` and `ols_bcm` require estimates of
the classifiers false-positive rate.

We estimate the false positive rate manually. To do so, we took a random
sample of size 1000 postings. Of these, 26 had `wfh_wham = 1`. based on
reading tese 26 postings, 9 appeared to be misclassified. This means the
estimated false-positive rate is 0.009. Accordingly, we will implement
`ols_bca` and `ols_bcm` with `fpr = 0.009`(the estimated false-positive
rate) and `m = 1000` (the sample size used to estimate the false
positive rate).

## Results

We first present results for a simple regression of log alary onto the
remote work indicator. We then consider a second specification with
fixed effects.

We compare standard OLS estimates and confidence intervals with
estimates and confidence intervals using `ols_bcm` which performs a
direct bias correction and computes bias corrected CIs, and `one_step`
which performs maximum likelihod estimation treating the true labels as
latent.

### Without fixed effects

We first present OLS estimates:

``` r
lm_SD_1 <- ols(salary ~ wfh_wham, data = SD_data)
summary(lm_SD_1)
```

    ## 
    ## MLBC Model Summary
    ## ==================
    ## 
    ## Formula:  Y ~ Beta_0 + Beta_1 * wfh_wham 
    ## 
    ## 
    ## Coefficients:
    ## 
    ##        Estimate Std.Error   z.value Pr(>|z|) Signif             95% CI
    ## Beta_0  10.6560    0.0026 4115.0944  < 2e-16    *** [10.6509, 10.6610]
    ## Beta_1   0.6485    0.0249   26.0334  < 2e-16    ***   [0.5997, 0.6973]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Now using the multiplicative bias correction, with bias corrected CIs:

``` r
SD_bcm_1 <- ols_bcm(salary ~ wfh_wham, data = SD_data, fpr = 0.009, m = 1000)
summary(SD_bcm_1)
```

    ## 
    ## MLBC Model Summary
    ## ==================
    ## 
    ## Formula:  Y ~ Beta_0 + Beta_1 * wfh_wham 
    ## 
    ## 
    ## Coefficients:
    ## 
    ##        Estimate Std.Error   z.value Pr(>|z|) Signif             95% CI
    ## Beta_0  10.6463    0.0042 2550.6119  < 2e-16    *** [10.6381, 10.6544]
    ## Beta_1   1.0524    0.1400    7.5156 5.67e-14    ***   [0.7780, 1.3269]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Finally, using maximum likelihood:

``` r
os_SD_1 <- one_step(salary ~ wfh_wham, data = SD_data)
summary(os_SD_1)
```

    ## 
    ## MLBC Model Summary
    ## ==================
    ## 
    ## Formula:  Y ~ Beta_0 + Beta_1 * wfh_wham 
    ## 
    ## Number of observations: 16315 
    ## Log-likelihood: -2475.893 
    ## 
    ## Coefficients:
    ## 
    ##        Estimate Std.Error   z.value Pr(>|z|) Signif             95% CI
    ## Beta_0  10.5264    0.0015 6840.2578  < 2e-16    *** [10.5234, 10.5295]
    ## Beta_1   0.4861    0.0084   57.7818  < 2e-16    ***   [0.4697, 0.5026]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Note that the one-step estimates here use the default standard normal
distribution for the regression errors. These estimates differ from
Table 1 of [Battaglia, Christensen, Hansen & Sacher
(2024)](https://arxiv.org/abs/2402.15585) which instead uses a Gaussian
mixture.

### With fixed effects

We repeat the above now with fixed effects, which are easily generated
for the categorical variables `soc_2021_2` amd `employment_type_name`.

First, using OLS:

``` r
lm_SD_2 <- ols(salary ~ wfh_wham + soc_2021_2 + employment_type_name, data = SD_data)
summary(lm_SD_2)
```

    ## 
    ## MLBC Model Summary
    ## ==================
    ## 
    ## Formula:  Y ~ Beta_0 + Beta_1 * wfh_wham + Beta_2 * soc_2021_213-0000 + Beta_3 * soc_2021_215-0000 + Beta_4 * soc_2021_217-0000 + Beta_5 * soc_2021_219-0000 + Beta_6 * soc_2021_221-0000 + Beta_7 * soc_2021_223-0000 + Beta_8 * soc_2021_225-0000 + Beta_9 * soc_2021_227-0000 + Beta_10 * soc_2021_229-0000 + Beta_11 * soc_2021_231-0000 + Beta_12 * soc_2021_233-0000 + Beta_13 * soc_2021_235-0000 + Beta_14 * soc_2021_237-0000 + Beta_15 * soc_2021_239-0000 + Beta_16 * soc_2021_241-0000 + Beta_17 * soc_2021_243-0000 + Beta_18 * soc_2021_245-0000 + Beta_19 * soc_2021_247-0000 + Beta_20 * soc_2021_249-0000 + Beta_21 * soc_2021_251-0000 + Beta_22 * soc_2021_253-0000 + Beta_23 * soc_2021_255-0000 + Beta_24 * soc_2021_299-0000 + Beta_25 * employment_type_namePart-time (≤ 32 hours) + Beta_26 * employment_type_namePart-time / full-time 
    ## 
    ## 
    ## Coefficients:
    ## 
    ##         Estimate Std.Error   z.value Pr(>|z|) Signif             95% CI
    ## Beta_0   11.0883    0.0090 1226.6801  < 2e-16    *** [11.0705, 11.1060]
    ## Beta_1    0.3639    0.0215   16.8923  < 2e-16    ***   [0.3217, 0.4061]
    ## Beta_2   -0.1435    0.0212   -6.7768 1.23e-11    *** [-0.1849, -0.1020]
    ## Beta_3    0.1970    0.0361    5.4489 5.07e-08    ***   [0.1261, 0.2678]
    ## Beta_4   -0.0461    0.0445   -1.0344   0.3009         [-0.1333, 0.0412]
    ## Beta_5   -0.0531    0.0720   -0.7376   0.4608         [-0.1942, 0.0880]
    ## Beta_6   -0.3044    0.0401   -7.5915 3.16e-14    *** [-0.3829, -0.2258]
    ## Beta_7   -0.0616    0.2976   -0.2070   0.8360         [-0.6449, 0.5217]
    ## Beta_8   -0.2241    0.0353   -6.3479 2.18e-10    *** [-0.2933, -0.1549]
    ## Beta_9   -0.2511    0.0405   -6.1984 5.70e-10    *** [-0.3305, -0.1717]
    ## Beta_10   0.1175    0.0638    1.8407   0.0657      .  [-0.0076, 0.2425]
    ## Beta_11  -0.4298    0.0284  -15.1354  < 2e-16    *** [-0.4854, -0.3741]
    ## Beta_12  -0.3882    0.0145  -26.7210  < 2e-16    *** [-0.4167, -0.3597]
    ## Beta_13  -0.4407    0.0100  -43.8756  < 2e-16    *** [-0.4603, -0.4210]
    ## Beta_14  -0.4679    0.0104  -45.0573  < 2e-16    *** [-0.4883, -0.4476]
    ## Beta_15  -0.4459    0.0186  -23.9241  < 2e-16    *** [-0.4825, -0.4094]
    ## Beta_16  -0.3862    0.0127  -30.3760  < 2e-16    *** [-0.4111, -0.3613]
    ## Beta_17  -0.4042    0.0105  -38.3762  < 2e-16    *** [-0.4249, -0.3836]
    ## Beta_18  -0.1559    0.0090  -17.2513  < 2e-16    *** [-0.1737, -0.1382]
    ## Beta_19  -0.2782    0.0305   -9.1189  < 2e-16    *** [-0.3379, -0.2184]
    ## Beta_20  -0.3279    0.0144  -22.7297  < 2e-16    *** [-0.3562, -0.2996]
    ## Beta_21  -0.4383    0.0131  -33.3474  < 2e-16    *** [-0.4640, -0.4125]
    ## Beta_22  -0.4287    0.0126  -33.8906  < 2e-16    *** [-0.4535, -0.4039]
    ## Beta_23  -0.5578    0.0090  -61.7048  < 2e-16    *** [-0.5755, -0.5400]
    ## Beta_24  -0.3577    0.0182  -19.6670  < 2e-16    *** [-0.3934, -0.3221]
    ## Beta_25  -0.1688    0.0054  -31.2136  < 2e-16    *** [-0.1794, -0.1582]
    ## Beta_26  -0.1610    0.0050  -32.4754  < 2e-16    *** [-0.1707, -0.1513]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Now using the multiplicative bias correction:

``` r
SD_bcm_2 <- ols_bcm(salary ~ wfh_wham + soc_2021_2 + employment_type_name, data = SD_data, fpr = 0.009, m = 1000)
summary(SD_bcm_2)
```

    ## 
    ## MLBC Model Summary
    ## ==================
    ## 
    ## Formula:  Y ~ Beta_0 + Beta_1 * wfh_wham + Beta_2 * soc_2021_213-0000 + Beta_3 * soc_2021_215-0000 + Beta_4 * soc_2021_217-0000 + Beta_5 * soc_2021_219-0000 + Beta_6 * soc_2021_221-0000 + Beta_7 * soc_2021_223-0000 + Beta_8 * soc_2021_225-0000 + Beta_9 * soc_2021_227-0000 + Beta_10 * soc_2021_229-0000 + Beta_11 * soc_2021_231-0000 + Beta_12 * soc_2021_233-0000 + Beta_13 * soc_2021_235-0000 + Beta_14 * soc_2021_237-0000 + Beta_15 * soc_2021_239-0000 + Beta_16 * soc_2021_241-0000 + Beta_17 * soc_2021_243-0000 + Beta_18 * soc_2021_245-0000 + Beta_19 * soc_2021_247-0000 + Beta_20 * soc_2021_249-0000 + Beta_21 * soc_2021_251-0000 + Beta_22 * soc_2021_253-0000 + Beta_23 * soc_2021_255-0000 + Beta_24 * soc_2021_299-0000 + Beta_25 * employment_type_namePart-time (≤ 32 hours) + Beta_26 * employment_type_namePart-time / full-time 
    ## 
    ## 
    ## Coefficients:
    ## 
    ##         Estimate Std.Error   z.value Pr(>|z|) Signif             95% CI
    ## Beta_0   11.0741    0.0103 1071.5564  < 2e-16    *** [11.0538, 11.0943]
    ## Beta_1    0.6413    0.0996    6.4382 1.21e-10    ***   [0.4461, 0.8365]
    ## Beta_2   -0.1832    0.0252   -7.2557 3.99e-13    *** [-0.2327, -0.1337]
    ## Beta_3    0.1028    0.0491    2.0932   0.0363      *   [0.0065, 0.1990]
    ## Beta_4   -0.0576    0.0447   -1.2874   0.1980         [-0.1452, 0.0301]
    ## Beta_5   -0.1048    0.0744   -1.4090   0.1588         [-0.2506, 0.0410]
    ## Beta_6   -0.3207    0.0404   -7.9321 2.16e-15    *** [-0.3999, -0.2415]
    ## Beta_7   -0.1267    0.2979   -0.4252   0.6707         [-0.7105, 0.4572]
    ## Beta_8   -0.2290    0.0353   -6.4787 9.25e-11    *** [-0.2982, -0.1597]
    ## Beta_9   -0.2737    0.0413   -6.6269 3.43e-11    *** [-0.3547, -0.1928]
    ## Beta_10   0.1206    0.0638    1.8891   0.0589      .  [-0.0045, 0.2456]
    ## Beta_11  -0.4176    0.0287  -14.5386  < 2e-16    *** [-0.4739, -0.3613]
    ## Beta_12  -0.3769    0.0151  -25.0284  < 2e-16    *** [-0.4065, -0.3474]
    ## Beta_13  -0.4295    0.0108  -39.7882  < 2e-16    *** [-0.4507, -0.4084]
    ## Beta_14  -0.4554    0.0113  -40.3304  < 2e-16    *** [-0.4775, -0.4333]
    ## Beta_15  -0.4333    0.0192  -22.6107  < 2e-16    *** [-0.4709, -0.3958]
    ## Beta_16  -0.3835    0.0128  -30.0699  < 2e-16    *** [-0.4085, -0.3585]
    ## Beta_17  -0.3991    0.0107  -37.3003  < 2e-16    *** [-0.4201, -0.3782]
    ## Beta_18  -0.1418    0.0103  -13.7164  < 2e-16    *** [-0.1620, -0.1215]
    ## Beta_19  -0.2644    0.0309   -8.5576  < 2e-16    *** [-0.3249, -0.2038]
    ## Beta_20  -0.3153    0.0151  -20.8789  < 2e-16    *** [-0.3449, -0.2857]
    ## Beta_21  -0.4284    0.0136  -31.4633  < 2e-16    *** [-0.4551, -0.4017]
    ## Beta_22  -0.4168    0.0133  -31.2717  < 2e-16    *** [-0.4430, -0.3907]
    ## Beta_23  -0.5436    0.0103  -52.5981  < 2e-16    *** [-0.5638, -0.5233]
    ## Beta_24  -0.3526    0.0183  -19.2737  < 2e-16    *** [-0.3884, -0.3167]
    ## Beta_25  -0.1647    0.0056  -29.4199  < 2e-16    *** [-0.1756, -0.1537]
    ## Beta_26  -0.1564    0.0052  -30.0662  < 2e-16    *** [-0.1666, -0.1462]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Comparing these results with the OLS results above, we see that the bias
corrected CI for the slope coefficient lies to the right of the OLS CI.

Finally, using maximum likelihood:

``` r
os_SD_2 <- one_step(salary ~ wfh_wham + soc_2021_2 + employment_type_name, data = SD_data)
```

    ## Warning in one_step.default(Y = y, Xhat = Xm, homoskedastic = homoskedastic, :
    ## Optimization may not have converged (code: 1)

``` r
summary(os_SD_2) 
```

    ## 
    ## MLBC Model Summary
    ## ==================
    ## 
    ## Formula:  Y ~ Beta_0 + Beta_1 * wfh_wham + Beta_2 * soc_2021_213-0000 + Beta_3 * soc_2021_215-0000 + Beta_4 * soc_2021_217-0000 + Beta_5 * soc_2021_219-0000 + Beta_6 * soc_2021_221-0000 + Beta_7 * soc_2021_223-0000 + Beta_8 * soc_2021_225-0000 + Beta_9 * soc_2021_227-0000 + Beta_10 * soc_2021_229-0000 + Beta_11 * soc_2021_231-0000 + Beta_12 * soc_2021_233-0000 + Beta_13 * soc_2021_235-0000 + Beta_14 * soc_2021_237-0000 + Beta_15 * soc_2021_239-0000 + Beta_16 * soc_2021_241-0000 + Beta_17 * soc_2021_243-0000 + Beta_18 * soc_2021_245-0000 + Beta_19 * soc_2021_247-0000 + Beta_20 * soc_2021_249-0000 + Beta_21 * soc_2021_251-0000 + Beta_22 * soc_2021_253-0000 + Beta_23 * soc_2021_255-0000 + Beta_24 * soc_2021_299-0000 + Beta_25 * employment_type_namePart-time (≤ 32 hours) + Beta_26 * employment_type_namePart-time / full-time 
    ## 
    ## Number of observations: 16315 
    ## Log-likelihood: -420.8325 
    ## Convergence code: 1 (check convergence!)
    ## 
    ## Coefficients:
    ## 
    ##         Estimate Std.Error  z.value Pr(>|z|) Signif             95% CI
    ## Beta_0   10.9504    0.0111 990.9090  < 2e-16    *** [10.9287, 10.9720]
    ## Beta_1    0.2915    0.0111  26.3626  < 2e-16    ***   [0.2698, 0.3131]
    ## Beta_2   -0.1480    0.0156  -9.4983  < 2e-16    *** [-0.1786, -0.1175]
    ## Beta_3    0.1612    0.0368   4.3846 1.16e-05    ***   [0.0891, 0.2332]
    ## Beta_4   -0.1535    0.0235  -6.5455 5.93e-11    *** [-0.1995, -0.1075]
    ## Beta_5   -0.2160    0.0393  -5.4948 3.91e-08    *** [-0.2930, -0.1389]
    ## Beta_6   -0.2685    0.0251 -10.6989  < 2e-16    *** [-0.3177, -0.2193]
    ## Beta_7   -0.0795    0.0854  -0.9302   0.3522         [-0.2470, 0.0880]
    ## Beta_8   -0.2643    0.0377  -7.0198 2.22e-12    *** [-0.3381, -0.1905]
    ## Beta_9   -0.3077    0.0292 -10.5466  < 2e-16    *** [-0.3649, -0.2505]
    ## Beta_10  -0.0836    0.0348  -2.4016   0.0163      * [-0.1518, -0.0154]
    ## Beta_11  -0.3684    0.0248 -14.8815  < 2e-16    *** [-0.4169, -0.3199]
    ## Beta_12  -0.2957    0.0148 -20.0452  < 2e-16    *** [-0.3246, -0.2668]
    ## Beta_13  -0.4046    0.0114 -35.4813  < 2e-16    *** [-0.4269, -0.3822]
    ## Beta_14  -0.3814    0.0117 -32.6061  < 2e-16    *** [-0.4043, -0.3584]
    ## Beta_15  -0.4165    0.0141 -29.4757  < 2e-16    *** [-0.4442, -0.3888]
    ## Beta_16  -0.3686    0.0122 -30.2011  < 2e-16    *** [-0.3925, -0.3446]
    ## Beta_17  -0.3422    0.0116 -29.4791  < 2e-16    *** [-0.3650, -0.3195]
    ## Beta_18  -0.0365    0.1304  -0.2802   0.7793         [-0.2922, 0.2191]
    ## Beta_19  -0.2099    0.0260  -8.0666 7.23e-16    *** [-0.2609, -0.1589]
    ## Beta_20  -0.2614    0.0144 -18.1948  < 2e-16    *** [-0.2895, -0.2332]
    ## Beta_21  -0.3779    0.0129 -29.2360  < 2e-16    *** [-0.4033, -0.3526]
    ## Beta_22  -0.3743    0.0133 -28.2042  < 2e-16    *** [-0.4003, -0.3483]
    ## Beta_23  -0.4384    0.1304  -3.3611   0.0008    *** [-0.6940, -0.1827]
    ## Beta_24  -0.3657    0.0138 -26.4273  < 2e-16    *** [-0.3929, -0.3386]
    ## Beta_25  -0.0856    0.0034 -25.5329  < 2e-16    *** [-0.0921, -0.0790]
    ## Beta_26  -0.0842    0.0034 -24.5756  < 2e-16    *** [-0.0909, -0.0775]
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

As before, the results for `one_step` diverge from those reported in the
“Joint” column of Table 1 in the paper as those were generated using a
Gaussian mixture model.
