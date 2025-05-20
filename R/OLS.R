#' Ordinary least squares (and heteroskedastic-robust SEs)
#'
#' Can be called either as
#'   * `ols(Y, X, se, intercept)` where `Y` is a numeric vector and `X` a matrix, or
#'   * `ols(formula, data, se, intercept)` using a one-sided formula like `y ~ x1 + x2`.
#'
#' @param Y numeric response vector, or a one-sided formula
#' @param X numeric design matrix (if `Y` is numeric)
#' @param data data frame (if `Y` is a formula)
#' @param se logical; return heteroskedasticity-robust SEs?
#' @param intercept logical; include an intercept term?
#' @param ... unused
#' @return An object of class `mlbc_fit` and `mlbc_ols` with elements:
#'   - `coef`: coefficient vector
#'   - `vcov`: variance–covariance matrix (if `se = TRUE`)
#'   - `sXX`: scaled crossproduct `X'X/n`
#' @export
ols <- function(Y, X = NULL, se = TRUE, intercept = FALSE, ...) {
  UseMethod("ols")
}

#' @rdname ols
#' @method ols default
#' @export
ols.default <- function(Y, X, se = TRUE, intercept = FALSE, ...) {
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  if (intercept) {
    X <- cbind(Intercept = 1, X)
  }
  n   <- nrow(X)
  sXX <- crossprod(X) / n
  sXY <- crossprod(X, Y) / n

  ## solve for coefficients via Cholesky
  C <- chol(sXX)
  b <- backsolve(C, forwardsolve(t(C), sXY))

  if (!se) {
    res <- list(coef = b, sXX = sXX)
    class(res) <- c("mlbc_fit", "mlbc_ols")
    return(res)
  }

  ## heteroskedasticity-robust sandwich
  u     <- as.vector(Y - X %*% b)
  Xu    <- X * u
  Omega <- crossprod(Xu)
  invXX <- chol2inv(C)
  V     <- invXX %*% Omega %*% invXX / (n^2)

  res <- list(coef = b, vcov = V, sXX = sXX)
  class(res) <- c("mlbc_fit", "mlbc_ols")
  res
}

#' @rdname ols
#' @method ols formula
#' @importFrom stats model.frame model.response model.matrix
#' @export
ols.formula <- function(Y, data = parent.frame(), se = TRUE, intercept = TRUE, ...) {
  mf <- model.frame(Y, data)
  y  <- model.response(mf)
  Xm <- model.matrix(Y, data)
  # model.matrix already includes an intercept column if needed,
  # so tell the default method not to add one
  ols.default(y, Xm, se = se, intercept = FALSE, ...)
}
