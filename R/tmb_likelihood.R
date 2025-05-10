#' Convert theta vector to mixture parameters
#' @noRd
theta_to_pars <- function(theta, d, homoskedastic) {
  b      <- theta[1:d]
  v_raw  <- theta[(d+1):(d+3)]
  w_raw  <- exp(v_raw)
  w      <- w_raw / (1 + sum(w_raw))
  sigma0 <- exp(theta[d+4])
  sigma1 <- if (homoskedastic) sigma0 else exp(theta[d+5])
  list(coef = b,
       weight0 = w[1], weight1 = w[2], weight2 = w[3],
       sigma0 = sigma0, sigma1 = sigma1)
}

#' Negative log-likelihood for unlabeled mixture
#' @noRd
likelihood_unlabeled <- function(Y, Xhat, theta, homoskedastic, distribution = NULL) {
  pars  <- theta_to_pars(theta, ncol(Xhat), homoskedastic)
  b     <- pars$coef
  w00   <- pars$weight0; w01 <- pars$weight1; w10 <- pars$weight2
  w11   <- 1 - (w00 + w01 + w10)
  mu    <- Xhat %*% b
  pdf   <- if (is.null(distribution)) dnorm else distribution
  term1 <- w11*pdf(Y, mu, pars$sigma1) + w10*pdf(Y, mu-b[1], pars$sigma0)
  term0 <- w01*pdf(Y, mu+b[1], pars$sigma1) + w00*pdf(Y, mu,     pars$sigma0)
  -sum(log(ifelse(Xhat[,1]==1, term1, term0)))
}
