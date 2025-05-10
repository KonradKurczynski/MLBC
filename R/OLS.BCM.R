ols_bcm <- function(Y, Xhat, fpr, m) {
  orig <- ols(Y, Xhat, se = TRUE)
  b0   <- as.numeric(orig$coef)
  V0   <- orig$vcov
  sXX  <- orig$sXX

  d    <- length(b0)
  A    <- matrix(0, nrow = d, ncol = d)
  A[1,1] <- 1
  Gamma <- solve(sXX, A)

  I  <- diag(d)
  b  <- solve(I - fpr * Gamma, b0)

  V1 <- solve(I - fpr * Gamma) %*% V0 %*% t(solve(I - fpr * Gamma))
  V2 <- (fpr * (1 - fpr) / m) *
    ( Gamma %*% (V0 + tcrossprod(b)) %*% t(Gamma) )
  V  <- V1 + V2

  list(coef = b, vcov = V)
}

