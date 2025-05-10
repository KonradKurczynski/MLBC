#' Compute starting values for TMB
#' @noRd
initial_guess <- function(Y, Xhat, homoskedastic) {
  ols_out <- ols(Y, Xhat, se = FALSE)
  b0      <- as.numeric(ols_out$coef)
  u       <- Y - Xhat %*% b0
  sigma   <- sd(u)
  mu      <- Xhat %*% b0
  pdf     <- function(y,loc,sc) dnorm(y,loc,sc)
  cond1   <- pdf(Y, mu, sigma) > pdf(Y, mu - b0[1], sigma)
  cond2   <- pdf(Y, mu + b0[1], sigma) > pdf(Y, mu, sigma)
  Ximp    <- ifelse(Xhat[,1]==1, cond1, cond2)
  w00     <- mean((Xhat[,1]==0)&(Ximp==0)); w01 <- mean((Xhat[,1]==0)&(Ximp==1))
  w10     <- mean((Xhat[,1]==1)&(Ximp==0))
  v_raw   <- log(c(max(w00,1e-3),max(w01,1e-3),max(w10,1e-3)) /
                   (1 - (w00+w01+w10)))
  sigma0  <- sd(u[Ximp==0]); sigma1 <- sd(u[Ximp==1])
  sigma0  <- ifelse(is.nan(sigma0), sigma1, sigma0)
  sigma1  <- ifelse(is.nan(sigma1), sigma0, sigma1)
  if (homoskedastic) {
    p_imp  <- mean(Ximp)
    sigma_c<- sigma1*p_imp + sigma0*(1-p_imp)
    c(b0, v_raw, log(sigma_c))
  } else {
    c(b0, v_raw, log(sigma0), log(sigma1))
  }
}

#' Subset standard deviation
#' @noRd
subset_std <- function(x, mask) sd(x[mask])
