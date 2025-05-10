#' Ordinary least squares (and heteroskedastic‐robust SEs)
#'
#' @param Y numeric response
#' @param X numeric design matrix
#' @param se  logical; return SEs?
#' @return list(coef, vcov, sXX) or list(coef, sXX)
#' @export

ols <- function(Y, X, se = TRUE) {
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  n <- nrow(X); d <- ncol(X)

  # 1/n–scaled cross‐products
  sXX <- crossprod(X) / n         # t(X) %*% X / n
  sXY <- crossprod(X, Y)   / n    # t(X) %*% Y / n

  # solve via Cholesky: sXX = C'C
  C <- chol(sXX)
  # b = solve(sXX, sXY)
  b <- backsolve(C, forwardsolve(t(C), sXY))

  if (!se) {
    return(list(coef = b, sXX = sXX))
  }

  # vectorized Omega = X' diag(u^2) X
  u     <- drop(Y - X %*% b)
  Xu    <- X * u                     # each row i scaled by u[i]
  Omega <- crossprod(Xu)             # = t(Xu) %*% Xu

  invXX <- chol2inv(C)               # = solve(sXX)
  V     <- invXX %*% Omega %*% invXX / (n^2)

  list(coef = b, vcov = V, sXX = sXX)
}
