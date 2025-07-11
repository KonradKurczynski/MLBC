#' @param coef Named numeric vector of coefficients.
#' @param vcov Variance–covariance matrix.
#' @param ml_name Name of the ML‐generated regressor (must match one `coef` name).
#' @return A list with elements `coef` and `vcov`, reordered.
#' @keywords internal
#' @noRd
.reorder_coefs <- function(coef, vcov, ml_name) {
  nm      <- names(coef)
  if (!ml_name %in% nm) {
    stop("ML variable '", ml_name, "' not found in coef names: ", paste(nm, collapse = ", "))
  }
  int_idx <- which(nm %in% c("(Intercept)", "Intercept"))
  slopes  <- setdiff(nm, nm[int_idx])
  perm <- c(ml_name, setdiff(slopes, ml_name), nm[int_idx])
  perm <- perm[perm %in% nm]
  list(
    coef = coef[perm],
    vcov = vcov[perm, perm, drop = FALSE]
  )
}

#' Additive bias-corrected OLS (BCA)
#'
#' Performs an additive bias correction to regressions that include a binary
#' covariate generated by AI/ML. This method requires an external estimate of
#' the false-positive rate. Standard errors are adjusted to account for
#' uncertainty in the false-positive rate estimate.
#'
#' @section Usage Options:
#'
#' **Option 1: Formula Interface**
#' - `Y`: A one-sided formula string
#' - `data`: Data frame containing the variables referenced in the formula
#'
#' **Option 2: Array Interface**
#' - `Y`: Response variable vector
#' - `Xhat`: Design matrix of covariates
#'
#' @param Y numeric response vector, or a one-sided formula
#' @param Xhat numeric matrix of regressors (if `Y` is numeric); the ML-regressor is column `gen_idx`
#' @param fpr numeric; estimated false-positive rate of the ML regressor
#' @param m integer; size of the external sample used to estimate the classifier's false-positive rate. Can be set to a large number when the false-positive rate is known exactly
#' @param data data frame (if `Y` is a formula)
#' @param intercept logical; if `TRUE`, prepends a column of 1's to `Xhat`
#' @param gen_idx integer; 1-based index of the ML-generated variable to apply bias correction to. If not specified, defaults to the first non-intercept variable
#' @param ... unused
#'
#' @return An object of class `mlbc_fit` and `mlbc_bca` with:
#'   - `coef`: bias-corrected coefficient estimates (ML-slope first, other slopes, intercept last)
#'   - `vcov`: adjusted variance-covariance matrix for those coefficients
#'
#' @examples
#' # Load the remote work dataset
#' data(SD_data)
#'
#' # Formula interface
#' fit_bca <- ols_bca(log(salary) ~ wfh_wham + soc_2021_2 + employment_type_name,
#'                    data = SD_data,
#'                    fpr = 0.009,  # estimated false positive rate
#'                    m = 1000)     # validation sample size
#' summary(fit_bca)
#'
#' # Array interface
#' Y <- log(SD_data$salary)
#' Xhat <- model.matrix(~ wfh_wham + soc_2021_2, data = SD_data)[, -1]
#' fit_bca2 <- ols_bca(Y, Xhat, fpr = 0.009, m = 1000, intercept = TRUE)
#' summary(fit_bca2)
#'
#' @export
ols_bca <- function(Y, Xhat = NULL, fpr, m, data = parent.frame(),
                    intercept = TRUE, gen_idx = 1, ...) {
  UseMethod("ols_bca")
}


#' @rdname ols_bca
#' @method ols_bca default
#' @export
ols_bca.default <- function(Y, Xhat, fpr, m, data = parent.frame(),
                            intercept = TRUE, gen_idx = 1, ...) {
  Y    <- as.numeric(Y)
  Xhat <- as.matrix(Xhat)

  if (intercept) {
    Xhat    <- cbind(Intercept = 1, Xhat)
    gen_idx <- gen_idx + 1L
  }

  ml_name <- colnames(Xhat)[gen_idx]

  n   <- nrow(Xhat)
  sXX <- crossprod(Xhat) / n
  sXY <- crossprod(Xhat, Y) / n

  C   <- chol(sXX)
  b0  <- backsolve(C, forwardsolve(t(C), sXY))

  u   <- as.vector(Y - Xhat %*% b0)
  Xu  <- Xhat * u
  Omega <- crossprod(Xu)
  invXX <- chol2inv(C)
  V0    <- invXX %*% Omega %*% invXX / (n^2)

  d <- length(b0)

  A     <- matrix(0, d, d)
  A[gen_idx, gen_idx] <- 1L
  Gamma <- solve(sXX, A)
  Minv  <- solve(diag(d) + fpr * Gamma)

  b_raw <- as.numeric(b0 + fpr * (Gamma %*% b0))
  V1    <- Minv %*% V0 %*% t(Minv)
  V2    <- (fpr * (1 - fpr) / m) *
    (Gamma %*% (V0 + tcrossprod(b0)) %*% t(Gamma))
  V_raw <- V1 + V2

  names(b_raw)            <- colnames(V_raw) <- rownames(V_raw) <- colnames(Xhat)
  out_coefs <- .reorder_coefs(b_raw, V_raw, ml_name)

  res <- list(coef = out_coefs$coef, vcov = out_coefs$vcov)
  class(res) <- c("mlbc_fit","mlbc_bca")
  res
}


#' @rdname ols_bca
#' @method ols_bca formula
#' @importFrom stats model.frame model.response model.matrix terms
#' @export
ols_bca.formula <- function(Y, Xhat = NULL, fpr, m, data = parent.frame(),
                            intercept = TRUE, gen_idx = 1, ...) {
  mf        <- stats::model.frame(Y, data)
  y         <- stats::model.response(mf)
  terms_obj <- stats::terms(mf)
  Xmat      <- stats::model.matrix(terms_obj, mf)

  if ("(Intercept)" %in% colnames(Xmat)) {
    Xmat <- Xmat[, setdiff(colnames(Xmat), "(Intercept)"), drop = FALSE]
  }

  rhs_terms <- attr(terms_obj, "term.labels")
  one       <- match(rhs_terms[1], colnames(Xmat))
  if (is.na(one)) {
    stop("Could not locate ML term '", rhs_terms[1], "' in design matrix.")
  }

  ols_bca.default(y, Xmat,
                  fpr       = fpr,
                  m         = m,
                  intercept = intercept,
                  gen_idx   = one,
                  ...)
}

#' Multiplicative bias-corrected OLS (BCM)
#'
#' Performs a multiplicative bias correction to regressions that include a binary
#' covariate generated by AI/ML. This method requires an external estimate of
#' the false-positive rate. Standard errors are adjusted to account for
#' uncertainty in the false-positive rate estimate.
#'
#' @section Usage Options:
#'
#' **Option 1: Formula Interface**
#' - `Y`: A one-sided formula string
#' - `data`: Data frame containing the variables referenced in the formula
#'
#' **Option 2: Array Interface**
#' - `Y`: Response variable vector
#' - `Xhat`: Design matrix of covariates
#'
#' @inheritParams ols_bca
#'
#' @return An object of class `mlbc_fit` and `mlbc_bcm` with:
#'   - `coef`: bias-corrected coefficient estimates (ML-slope first, other slopes, intercept last)
#'   - `vcov`: adjusted variance-covariance matrix for those coefficients
#'
#' @examples
#' # Load the remote work dataset
#' data(SD_data)
#'
#' # Formula interface
#' fit_bcm <- ols_bcm(log(salary) ~ wfh_wham + soc_2021_2 + employment_type_name,
#'                    data = SD_data,
#'                    fpr = 0.009,  # estimated false positive rate
#'                    m = 1000)     # validation sample size
#' summary(fit_bcm)
#'
#' # Compare with uncorrected OLS
#' fit_ols <- ols(log(salary) ~ wfh_wham + soc_2021_2 + employment_type_name,
#'                data = SD_data)
#'
#' # Display coefficient comparison
#' data.frame(
#'   OLS = coef(fit_ols)[1:2],
#'   BCM = coef(fit_bcm)[1:2]
#' )
#'
#' @export
ols_bcm <- function(Y, Xhat = NULL, fpr, m, data = parent.frame(),
                    intercept = TRUE, gen_idx = 1, ...) {
  UseMethod("ols_bcm")
}

#' @rdname ols_bcm
#' @method ols_bcm default
#' @export
ols_bcm.default <- function(Y, Xhat, fpr, m, data = parent.frame(),
                            intercept = TRUE, gen_idx = 1, ...) {
  Y    <- as.numeric(Y)
  Xhat <- as.matrix(Xhat)

  if (intercept) {
    Xhat    <- cbind(Intercept = 1, Xhat)
    gen_idx <- gen_idx + 1L
  }

  ml_name <- colnames(Xhat)[gen_idx]

  n   <- nrow(Xhat)
  sXX <- crossprod(Xhat) / n
  sXY <- crossprod(Xhat, Y) / n

  C   <- chol(sXX)
  b0  <- backsolve(C, forwardsolve(t(C), sXY))

  u     <- as.vector(Y - Xhat %*% b0)
  Xu    <- Xhat * u
  Omega <- crossprod(Xu)
  invXX <- chol2inv(C)
  V0    <- invXX %*% Omega %*% invXX / (n^2)

  d <- length(b0)

  A     <- matrix(0, d, d); A[gen_idx, gen_idx] <- 1L
  Gamma <- solve(sXX, A)
  Minv  <- solve(diag(d) - fpr * Gamma)

  b_raw <- as.numeric(Minv %*% b0)
  V1    <- Minv %*% V0 %*% t(Minv)
  V2    <- (fpr * (1 - fpr) / m) *
    (Gamma %*% (V0 + tcrossprod(b_raw)) %*% t(Gamma))
  V_raw <- V1 + V2

  names(b_raw)            <- colnames(V_raw) <- rownames(V_raw) <- colnames(Xhat)
  out_coefs <- .reorder_coefs(b_raw, V_raw, ml_name)

  res <- list(coef = out_coefs$coef, vcov = out_coefs$vcov)
  class(res) <- c("mlbc_fit","mlbc_bcm")
  res
}

#' @rdname ols_bcm
#' @method ols_bcm formula
#' @importFrom stats model.frame model.response model.matrix terms
#' @export
ols_bcm.formula <- function(Y, Xhat = NULL, fpr, m, data = parent.frame(),
                            intercept = TRUE, gen_idx = 1, ...) {
  mf        <- stats::model.frame(Y, data)
  y         <- stats::model.response(mf)
  terms_obj <- stats::terms(mf)
  Xmat      <- stats::model.matrix(terms_obj, mf)

  if ("(Intercept)" %in% colnames(Xmat)) {
    Xmat <- Xmat[, setdiff(colnames(Xmat), "(Intercept)"), drop = FALSE]
  }

  rhs_terms <- attr(terms_obj, "term.labels")
  one       <- match(rhs_terms[1], colnames(Xmat))
  if (is.na(one)) {
    stop("Could not locate ML term '", rhs_terms[1], "' in design matrix.")
  }

  ols_bcm.default(y, Xmat,
                  fpr       = fpr,
                  m         = m,
                  intercept = intercept,
                  gen_idx   = one,
                  ...)
}
