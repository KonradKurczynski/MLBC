#' One-step maximum likelihood estimation
#'
#' Maximum likelihood estimation of the regression model, treating the generated
#' covariate as a noisy proxy for the true latent variable. This method is
#' particularly useful when an estimate of the false positive rate is not available.
#' The variance of the estimates is approximated via the inverse Hessian at the optimum.
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
#' @param Xhat numeric matrix of regressors (if `Y` is numeric)
#' @param homoskedastic logical; if `TRUE`, assumes a common error variance; otherwise, the error variance is allowed to vary with the true latent binary variable
#' @param distribution character; distribution for error terms. One of `"normal"`, `"t"`, `"laplace"`, `"gamma"`, `"beta"`
#' @param nu numeric; degrees of freedom (for Student-t distribution)
#' @param gshape numeric; shape parameter (for Gamma distribution)
#' @param gscale numeric; scale parameter (for Gamma distribution)
#' @param ba numeric; alpha parameter (for Beta distribution)
#' @param bb numeric; beta parameter (for Beta distribution)
#' @param intercept logical; if `TRUE`, prepend an intercept column to `Xhat`
#' @param gen_idx integer; index (1-based) of the binary ML-generated variable. If not specified, defaults to the first non-intercept variable
#' @param data data frame (if `Y` is a formula)
#' @param ... unused
#'
#' @return An object of class `mlbc_fit` and `mlbc_onestep` with:
#'   - `coef`: estimated regression coefficients
#'   - `vcov`: variance-covariance matrix
#'
#' @examples
#' # Load the remote work dataset
#' data(SD_data)
#'
#' # Basic one-step estimation
#' fit_onestep <- one_step(log(salary) ~ wfh_wham + soc_2021_2 + employment_type_name,
#'                         data = SD_data)
#' summary(fit_onestep)
#'
#' # With different error distribution
#' fit_t <- one_step(log(salary) ~ wfh_wham + soc_2021_2,
#'                   data = SD_data,
#'                   distribution = "t",
#'                   nu = 4)
#' summary(fit_t)
#'
#' # Homoskedastic errors
#' fit_homo <- one_step(log(salary) ~ wfh_wham + soc_2021_2,
#'                      data = SD_data,
#'                      homoskedastic = TRUE)
#' summary(fit_homo)
#'
#' @export
one_step <- function(Y,
                     Xhat = NULL,
                     homoskedastic = FALSE,
                     distribution  = c("normal","t","laplace","gamma","beta"),
                     nu            = 4,
                     gshape        = 2, gscale = 1,
                     ba            = 2, bb   = 2,
                     intercept     = TRUE,
                     gen_idx       = 1,
                     data          = parent.frame(),
                     ...) {
  UseMethod("one_step")
}

#' @rdname one_step
#' @method one_step default
#' @importFrom TMB MakeADFun
#' @export
one_step.default <- function(Y,
                             Xhat,
                             homoskedastic = FALSE,
                             distribution  = c("normal","t","laplace","gamma","beta"),
                             nu            = 4,
                             gshape        = 2, gscale = 1,
                             ba            = 2, bb   = 2,
                             intercept     = TRUE,
                             gen_idx       = 1,
                             ...) {
  Y    <- as.numeric(Y)
  Xhat <- as.matrix(Xhat)

  original_names <- colnames(Xhat)

  if (intercept) {
    Xhat    <- cbind(Intercept = 1, Xhat)
    gen_idx <- gen_idx + 1L
  }

  ml_name <- colnames(Xhat)[gen_idx]

  Xhat    <- Xhat[, c(gen_idx, setdiff(seq_len(ncol(Xhat)), gen_idx)), drop = FALSE]
  gen_idx <- 1L


  distribution <- match.arg(distribution)
  dist_code   <- switch(distribution,
                        normal  = 1L,
                        t       = 2L,
                        laplace = 3L,
                        gamma   = 4L,
                        beta    = 5L)

  data_list <- list(
    Y            = Y,
    Xhat         = Xhat,
    homoskedastic= as.integer(homoskedastic),
    dist_code    = dist_code,
    nu           = nu,
    gshape       = gshape,
    gscale       = gscale,
    ba           = ba,
    bb           = bb
  )

  theta_init <- initial_guess(Y, Xhat, homoskedastic)

  obj <- MakeADFun(
    data       = data_list,
    parameters = list(theta = theta_init),
    DLL        = "MLBC",
    silent     = TRUE
  )


  opt <- tryCatch({
    nlminb(
      start     = obj$par,
      objective = obj$fn,
      gradient  = obj$gr,
      control   = list(iter.max = 500, abs.tol = 1e-12, rel.tol = 1e-10)
    )
  }, error = function(e) {
    warning("Primary optimization failed, trying fallback method")
    optim(
      par = obj$par,
      fn = obj$fn,
      gr = obj$gr,
      method = "BFGS",
      control = list(maxit = 500, reltol = 1e-10)
    )
  })


  if (exists("convergence", opt) && opt$convergence != 0) {
    warning("Optimization may not have converged (code: ", opt$convergence, ")")
  }


  H <- tryCatch({
    obj$he(opt$par)
  }, error = function(e) {
    warning("Hessian computation failed, using numerical approximation")

    numDeriv::hessian(obj$fn, opt$par)
  })


  if (any(!is.finite(H))) {
    warning("Hessian contains non-finite values")
    H[!is.finite(H)] <- 0
  }


  eig_vals <- eigen(H, symmetric = TRUE, only.values = TRUE)$values
  min_eig <- min(eig_vals)

  if (min_eig <= 1e-12) {
    warning("Hessian is not positive definite (min eigenvalue: ", min_eig, "), adding regularization")
    H <- H + diag(abs(min_eig) + 1e-8, nrow(H))
  }


  d <- ncol(Xhat)
  b_raw <- opt$par[1:d]

  V_raw <- tryCatch({
    chol_H <- chol(H)
    V_full <- chol2inv(chol_H)
    V_full[1:d, 1:d, drop = FALSE]
  }, error = function(e) {
    tryCatch({

      solve(H)[1:d, 1:d, drop = FALSE]
    }, error = function(e2) {
      tryCatch({

        MASS::ginv(H)[1:d, 1:d, drop = FALSE]
      }, error = function(e3) {
        warning("All covariance matrix computations failed, returning diagonal matrix")
        diag(1e-4, d)
      })
    })
  })

  if (any(!is.finite(V_raw))) {
    warning("Covariance matrix contains non-finite values")
    V_raw[!is.finite(V_raw)] <- 0
    diag(V_raw) <- pmax(diag(V_raw), 1e-8)
  }

  ses <- sqrt(diag(V_raw))
  if (any(ses < 1e-10)) {
    warning("Some standard errors are extremely small, possible numerical issues")
    small_idx <- which(ses < 1e-10)
    diag(V_raw)[small_idx] <- 1e-8
  }

  names(b_raw) <- colnames(Xhat)
  colnames(V_raw) <- rownames(V_raw) <- colnames(Xhat)

  out_coefs <- .reorder_coefs(b_raw, V_raw, ml_name)

  out <- list(
    coef = out_coefs$coef,
    vcov = out_coefs$vcov,
    convergence = if(exists("convergence", opt)) opt$convergence else 0,
    loglik = -opt$objective,
    nobs = length(Y)
  )
  class(out) <- c("mlbc_fit", "mlbc_onestep")
  out
}

#' @rdname one_step
#' @method one_step formula
#' @importFrom stats model.frame model.response model.matrix
#' @export
one_step.formula <- function(Y,
                             Xhat = NULL,
                             homoskedastic = FALSE,
                             distribution  = c("normal","t","laplace","gamma","beta"),
                             nu            = 4,
                             gshape        = 2, gscale = 1,
                             ba            = 2, bb   = 2,
                             intercept     = TRUE,
                             gen_idx       = 1,
                             data          = parent.frame(),
                             ...) {

  mf <- stats::model.frame(Y, data = data)
  y  <- stats::model.response(mf)

  Xm <- stats::model.matrix(Y, data = data)

  if ("(Intercept)" %in% colnames(Xm)) {
    Xm <- Xm[, setdiff(colnames(Xm), "(Intercept)"), drop = FALSE]
  }

  rhs_terms <- attr(stats::terms(Y, data = data), "term.labels")
  ml_name   <- rhs_terms[1]
  gen_idx   <- match(ml_name, colnames(Xm))
  if (is.na(gen_idx)) {
    stop("Couldn't find ML-variable '", ml_name, "' in the design matrix.")
  }

  one_step.default(
    Y             = y,
    Xhat          = Xm,
    homoskedastic = homoskedastic,
    distribution  = distribution,
    nu            = nu,
    gshape        = gshape,
    gscale        = gscale,
    ba            = ba,
    bb            = bb,
    intercept     = intercept,
    gen_idx       = gen_idx,
    ...
  )
}


