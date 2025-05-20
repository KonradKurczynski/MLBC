#' Ordinary least squares (and heteroskedastic‐robust SEs)
#'
#' @param Y numeric response
#' @param X numeric design matrix
#' @param se  logical; return SEs?
#' @param intercept logical; include an intercept term in estimation?
#' @return list(coef, vcov, sXX) or list(coef, sXX)
#' @export

ols <- function(Y, X, se = TRUE, intercept = FALSE) {
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  if (intercept) {
    X <- cbind(Intercept = 1, X)
  }
  n <- nrow(X)

  sXX <- crossprod(X) / n
  sXY <- crossprod(X, Y) / n

  C <- chol(sXX)
  b <- backsolve(C, forwardsolve(t(C), sXY))

  if (!se) {
    res <- list(coef = b, sXX = sXX)
    class(res) <- c("mlbc_fit", "mlbc_ols")
    return(res)
  }

  u     <- Y - X %*% b
  Xu    <- X * as.vector(u)
  Omega <- crossprod(Xu)
  invXX <- chol2inv(C)
  V     <- invXX %*% Omega %*% invXX / (n^2)

  res <- list(coef = b, vcov = V, sXX = sXX)
  class(res) <- c("mlbc_fit", "mlbc_ols")
  res
}
