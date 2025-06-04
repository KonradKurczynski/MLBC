#' Ultra-minimal version
#' @param Y Response vector
#' @param Xhat Design matrix
#' @param homoskedastic Logical, whether to assume homoskedastic errors
#' @return Vector of starting parameter values
#' @noRd
initial_guess <- function(Y, Xhat, homoskedastic) {

  # OLS estimation
  b <- solve(crossprod(Xhat), crossprod(Xhat, Y))

  # Residuals and sigma
  u <- Y - Xhat %*% b
  sigma <- sd(u)

  # PDF comparisons for imputation
  mu <- Xhat %*% b
  X_obs <- Xhat[,1]

  # Vectorized PDF calculations
  pdf_current <- dnorm(Y, mu, sigma)
  pdf_alt1 <- dnorm(Y, mu - b[1], sigma)
  pdf_alt2 <- dnorm(Y, mu + b[1], sigma)

  # Imputation
  X_imp <- ifelse(X_obs == 1,
                  as.numeric(pdf_current > pdf_alt1),
                  as.numeric(pdf_alt2 > pdf_current))

  # Frequencies with minimal bounds
  freqs <- c(
    mean((X_obs == 0) & (X_imp == 0)),
    mean((X_obs == 0) & (X_imp == 1)),
    mean((X_obs == 1) & (X_imp == 0)),
    mean((X_obs == 1) & (X_imp == 1))
  )
  freqs <- pmax(freqs, 0.001)
  freqs <- freqs / sum(freqs)

  # Log-ratios
  v <- log(freqs[1:3] / freqs[4])

  # Component sigmas
  sigma0 <- sd(u[X_imp == 0])
  sigma1 <- sd(u[X_imp == 1])

  # Simple NaN handling
  if (is.na(sigma0)) sigma0 <- sigma
  if (is.na(sigma1)) sigma1 <- sigma

  # Return result
  if (homoskedastic) {
    p <- mean(X_imp)
    sigma_comb <- sigma1 * p + sigma0 * (1 - p)
    c(b, v, log(sigma_comb))
  } else {
    c(b, v, log(sigma0), log(sigma1))
  }
}
