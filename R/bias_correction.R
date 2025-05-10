#' Multiplicative bias correction
#' @export
ols_bcm <- function(Y, Xhat, fpr, m) {
  orig <- ols(Y, Xhat, se = TRUE)
  b0   <- as.numeric(orig$coef)
  V0   <- orig$vcov
  sXX  <- orig$sXX
  d    <- length(b0)

  A     <- matrix(0, nrow = d, ncol = d); A[1,1] <- 1
  Gamma <- solve(sXX, A)

  I    <- diag(d)
  Minv <- solve(I - fpr * Gamma)      # cache this inversion
  b    <- Minv %*% b0

  V1   <- Minv %*% V0 %*% t(Minv)
  V2   <- (fpr * (1 - fpr) / m) * (Gamma %*% (V0 + tcrossprod(b)) %*% t(Gamma))
  V    <- V1 + V2

  list(coef = b, vcov = V)
}

#' Additive bias correction
#' @export
ols_bca <- function(Y, Xhat, fpr, m) {
  orig <- ols(Y, Xhat, se = TRUE)
  b0   <- as.numeric(orig$coef)
  V0   <- orig$vcov
  sXX  <- orig$sXX
  d    <- length(b0)

  A     <- matrix(0, nrow = d, ncol = d); A[1,1] <- 1
  Gamma <- solve(sXX, A)

  b     <- b0 + fpr * (Gamma %*% b0)

  I     <- diag(d)
  Minv  <- solve(I + fpr * Gamma)    # cache inversion
  V1    <- Minv %*% V0 %*% t(Minv)
  V2    <- (fpr * (1 - fpr) / m) * (Gamma %*% (V0 + tcrossprod(b0)) %*% t(Gamma))
  V     <- V1 + V2

  list(coef = b, vcov = V)
}
