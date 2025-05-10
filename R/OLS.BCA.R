ols_bca <- function(Y, Xhat, fpr, m) {
  orig <- ols(Y, Xhat, se = TRUE)
  b0   <- as.numeric(orig$coef)
  V0   <- orig$vcov
  sXX  <- orig$sXX

  d    <- length(b0)

  A    <- matrix(0, nrow = d, ncol = d)
  A[1,1] <- 1

  Gamma <- solve(sXX, A)

  b <- b0 + fpr * (Gamma %*% b0)

  I  <- diag(d)
  V1 <- (I + fpr * Gamma) %*% V0 %*% t(I + fpr * Gamma)
  V2 <- (fpr * (1 - fpr) / m) *
    ( Gamma %*% (V0 + tcrossprod(b0)) %*% t(Gamma) )
  V  <- V1 + V2

  list(coef = b, vcov = V)
}

