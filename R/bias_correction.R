#' Additive bias‐corrected OLS estimator (BCA)
#'
#' Computes the additive bias correction for an OLS regression when the
#' primary regressor (column `gen_idx` of `Xhat`) is measured by an ML/AI method.
#'
#' @param Y numeric response vector, or a one‐sided formula `y ~ x1 + x2`
#' @param Xhat numeric matrix of regressors (if `Y` is numeric), or ignored if `Y` is a formula
#' @param fpr numeric; estimated false‐positive rate of the ML‐generated regressor
#' @param m integer; size of the labeled subsample used to estimate `fpr`
#' @param intercept logical; if `TRUE`, prepend a column of 1’s to `Xhat`
#' @param gen_idx integer; index (1-based) of the ML‐generated column in `Xhat` **before** adding an intercept
#' @param ... unused
#' @return An object of class `mlbc_fit` and `mlbc_bca` with `coef` and `vcov`
#' @export
ols_bca <- function(Y, Xhat = NULL, fpr, m, intercept = TRUE, gen_idx = 1, ...) {
  UseMethod("ols_bca")
}

#' @rdname ols_bca
#' @method ols_bca default
#' @export
ols_bca.default <- function(Y, Xhat, fpr, m, intercept = TRUE, gen_idx = 1, ...) {
  Y    <- as.numeric(Y)
  Xhat <- as.matrix(Xhat)
  if (intercept) {
    Xhat <- cbind(Intercept = 1, Xhat)
    gen_idx <- gen_idx + 1L
  }
  orig <- ols(Y, Xhat, se = TRUE, intercept = FALSE)
  b0   <- as.numeric(orig$coef)
  V0   <- orig$vcov
  sXX  <- orig$sXX
  d    <- length(b0)

  A         <- matrix(0, d, d)
  A[gen_idx, gen_idx] <- 1
  Gamma     <- solve(sXX, A)
  I         <- diag(d)
  Minv      <- solve(I + fpr * Gamma)

  b_raw <- b0 + fpr * (Gamma %*% b0)
  V1    <- Minv %*% V0 %*% t(Minv)
  V2    <- (fpr * (1 - fpr) / m) *
    (Gamma %*% (V0 + tcrossprod(b0)) %*% t(Gamma))
  V_raw <- V1 + V2

  if (intercept) {
    perm <- c(gen_idx, setdiff(seq_len(d), gen_idx))
    b    <- as.numeric(b_raw[perm])
    V    <- V_raw[perm, perm]
    names(b) <- colnames(Xhat)[perm]
  } else {
    b    <- as.numeric(b_raw)
    V    <- V_raw
    names(b) <- colnames(Xhat)
  }

  out <- list(coef = b, vcov = V)
  class(out) <- c("mlbc_fit", "mlbc_bca")
  out
}

#' @rdname ols_bca
#' @method ols_bca formula
#' @importFrom stats model.frame model.response model.matrix
#' @export
ols_bca.formula <- function(Y, Xhat = NULL, data = parent.frame(),
                            fpr, m, intercept = TRUE, gen_idx = 1, ...) {
  mf   <- model.frame(Y, data)
  y    <- model.response(mf)
  Xmat <- model.matrix(Y, data)
  # model.matrix already handles intercept/“-1”
  ols_bca.default(y, Xmat, fpr = fpr, m = m,
                  intercept = FALSE, gen_idx = gen_idx, ...)
}

#' Multiplicative bias‐corrected OLS estimator (BCM)
#'
#' Computes the multiplicative bias correction for an OLS regression when the
#' primary regressor (column `gen_idx` of `Xhat`) is measured by an ML/AI method.
#'
#' @param Y numeric response vector, or a one‐sided formula `y ~ x1 + x2`
#' @param Xhat numeric matrix of regressors (if `Y` is numeric), or ignored if `Y` is a formula
#' @param fpr numeric; estimated false‐positive rate of the ML‐generated regressor
#' @param m integer; size of the labeled subsample used to estimate `fpr`
#' @param intercept logical; if `TRUE`, prepend a column of 1’s to `Xhat`
#' @param gen_idx integer; index (1-based) of the ML‐generated column in `Xhat` **before** adding an intercept
#' @param ... unused
#' @return An object of class `mlbc_fit` and `mlbc_bcm` with `coef` and `vcov`
#' @export
ols_bcm <- function(Y, Xhat = NULL, fpr, m, intercept = TRUE, gen_idx = 1, ...) {
  UseMethod("ols_bcm")
}

#' @rdname ols_bcm
#' @method ols_bcm default
#' @export
ols_bcm.default <- function(Y, Xhat, fpr, m, intercept = TRUE,
                            gen_idx = 1, ...) {
  Y    <- as.numeric(Y)
  Xhat <- as.matrix(Xhat)
  if (intercept) {
    Xhat <- cbind(Intercept = 1, Xhat)
    gen_idx <- gen_idx + 1L
  }
  orig <- ols(Y, Xhat, se = TRUE, intercept = FALSE)
  b0   <- as.numeric(orig$coef)
  V0   <- orig$vcov
  sXX  <- orig$sXX
  d    <- length(b0)

  A       <- matrix(0, d, d)
  A[gen_idx, gen_idx] <- 1
  Gamma   <- solve(sXX, A)
  I       <- diag(d)
  Minv    <- solve(I - fpr * Gamma)

  b_raw <- Minv %*% b0
  V1    <- Minv %*% V0 %*% t(Minv)
  V2    <- (fpr * (1 - fpr) / m) *
    (Gamma %*% (V0 + tcrossprod(b_raw)) %*% t(Gamma))
  V_raw <- V1 + V2

  if (intercept) {
    perm <- c(gen_idx, setdiff(seq_len(d), gen_idx))
    b    <- as.numeric(b_raw[perm])
    V    <- V_raw[perm, perm]
    names(b) <- colnames(Xhat)[perm]
  } else {
    b    <- as.numeric(b_raw)
    V    <- V_raw
    names(b) <- colnames(Xhat)
  }

  out <- list(coef = b, vcov = V)
  class(out) <- c("mlbc_fit", "mlbc_bcm")
  out
}

#' @rdname ols_bcm
#' @method ols_bcm formula
#' @importFrom stats model.frame model.response model.matrix
#' @export
ols_bcm.formula <- function(Y, Xhat = NULL, data = parent.frame(),
                            fpr, m, intercept = TRUE, gen_idx = 1, ...) {
  mf   <- model.frame(Y, data)
  y    <- model.response(mf)
  Xmat <- model.matrix(Y, data)
  ols_bcm.default(y, Xmat,
                  fpr       = fpr,
                  m         = m,
                  intercept = FALSE,
                  gen_idx   = gen_idx, ...)
}

.reorder_intercept <- function(coef, vcov) {
  nm   <- names(coef)
  # find the intercept
  ints <- match("(Intercept)", nm, nomatch = 0L)
  if (ints == 0L) {
    # no intercept: return as-is
    return(list(coef = coef, vcov = vcov))
  }
  # build permutation: intercept first, then all others
  perm <- c(ints, seq_along(coef)[-ints])
  coef_new <- coef[perm]
  vcov_new <- vcov[perm, perm, drop = FALSE]
  names(coef_new) <- colnames(vcov_new) <- nm[perm]
  list(coef = coef_new, vcov = vcov_new)
}
