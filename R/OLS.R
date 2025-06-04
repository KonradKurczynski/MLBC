#' Ordinary Least Squares (OLS) regression
#'
#' Ordinary Least Squares regression with support for both formula and array-based
#' interfaces. This function provides a unified interface for fitting linear models
#' using either R formulas with data frames or raw matrices.
#'
#' @section Usage Options:
#'
#' **Option 1: Formula Interface**
#' - `Y`: A one-sided formula (e.g., `y ~ x1 + x2`)
#' - `data`: A data frame containing the variables referenced in the formula
#'
#' **Option 2: Array Interface**
#' - `Y`: Response variable vector
#' - `X`: Design matrix of covariates
#'
#' @param Y numeric response vector, or a one-sided formula
#' @param X numeric design matrix (if `Y` is numeric)
#' @param data data frame (if `Y` is a formula)
#' @param se logical; return heteroskedastic-robust standard errors?
#' @param intercept logical; include an intercept term?
#' @param ... unused
#'
#' @return An object of class `mlbc_fit` and `mlbc_ols` with:
#'   - `coef`: coefficient estimates
#'   - `vcov`: variance-covariance matrix
#'   - `sXX`: scaled cross-product X'X / n
#'
#' @examples
#' # Load the remote work dataset
#' data(SD_data)
#'
#' # Formula interface
#' fit1 <- ols(log(salary) ~ wfh_wham + soc_2021_2 + employment_type_name,
#'             data = SD_data)
#' summary(fit1)
#'
#' # Array interface
#' Y <- log(SD_data$salary)
#' X <- model.matrix(~ wfh_wham + soc_2021_2, data = SD_data)
#' fit2 <- ols(Y, X[, -1], intercept = TRUE)  # exclude intercept column
#' summary(fit2)
#'
#' @export
ols <- function(Y, X = NULL, data = parent.frame(), se = TRUE, intercept = FALSE, ...) {
  UseMethod("ols")
}

#-- default method: numeric Y + matrix X ---------------
#' @rdname ols
#' @method ols default
#' @export
ols.default <- function(Y, X, data = parent.frame(), se = TRUE, intercept = FALSE, ...){
  X <- as.matrix(X)
  Y <- as.numeric(Y)

  if (intercept) {
    X <- cbind(Intercept = 1, X)
  }

  n   <- nrow(X)
  sXX <- crossprod(X) / n
  sXY <- crossprod(X, Y) / n

  # solve for b
  C <- chol(sXX)
  b <- backsolve(C, forwardsolve(t(C), sXY))

  # compute robust V if requested
  if (se) {
    u     <- as.vector(Y - X %*% b)
    Xu    <- X * u
    Omega <- crossprod(Xu)
    invXX <- chol2inv(C)
    V     <- invXX %*% Omega %*% invXX / (n^2)
  } else {
    V <- NULL
  }

  names(b) <- colnames(X)

  int_pos <- which(names(b) %in% c("(Intercept)", "Intercept"))
  if (length(int_pos) == 1L) {
    perm <- c(seq_along(b)[-int_pos], int_pos)
    b    <- b[perm]
    if (se) V <- V[perm, perm, drop = FALSE]
  }

  res <- list(coef = b, vcov = V, sXX = sXX)
  class(res) <- c("mlbc_fit", "mlbc_ols")
  res
}

#-- formula method: y ~ x1 + x2  ----------------------
#' @rdname ols
#' @method ols formula
#' @importFrom stats model.frame model.response model.matrix
#' @export
ols.formula <- function(Y, X = NULL, data = parent.frame(), se = TRUE, intercept = TRUE, ...) {
  mf <- stats::model.frame(Y, data)
  y  <- stats::model.response(mf)
  Xm <- stats::model.matrix(Y, data)

  ols.default(y, Xm, se = se, intercept = FALSE, ...)
}
