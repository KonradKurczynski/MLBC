#' One-step estimator for unlabeled data (multi-dist)
#'
#' Fits the one-step estimator by maximizing the unlabeled likelihood via TMB,
#' automatically differentiating the objective, gradient, and Hessian.
#'
#' Can be called either as
#'  - `one_step(Y, Xhat, ...)` where `Y` is numeric and `Xhat` a matrix, or
#'  - `one_step(formula, data, ...)` using a one-sided formula like `y ~ x1 + x2`.
#'
#' @param Y numeric response vector, or a one-sided formula
#' @param Xhat numeric matrix of regressors (if `Y` is numeric)
#' @param homoskedastic logical; if `TRUE`, assume equal error variance
#' @param distribution character; one of `"normal"`, `"t"`, `"laplace"`, `"gamma"`, `"beta"`
#' @param nu numeric; degrees of freedom (for Student-t)
#' @param gshape numeric; shape (for Gamma)
#' @param gscale numeric; scale (for Gamma)
#' @param ba numeric; alpha (for Beta)
#' @param bb numeric; beta (for Beta)
#' @param intercept logical; if `TRUE`, prepend an intercept column to `Xhat`
#' @param gen_idx integer; index (1-based) of the ML-generated column in `Xhat` **before** intercept
#' @param data data frame (if `Y` is a formula)
#' @param ... unused
#' @return An object of class `mlbc_fit` and `mlbc_onestep` with:
#'   - `coef`: named coefficient vector
#'   - `cov` : covariance matrix
#' @importFrom TMB MakeADFun
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
  # reconcile intercept and gen_idx
  if (intercept) {
    Xhat    <- cbind(Intercept = 1, Xhat)
    gen_idx <- gen_idx + 1L
  }

  Xhat    <- Xhat[, c(gen_idx, setdiff(seq_len(ncol(Xhat)), gen_idx)), drop = FALSE]
  gen_idx <- 1L

  # map distribution to code
  distribution <- match.arg(distribution)
  dist_code   <- switch(distribution,
                        normal  = 1L,
                        t       = 2L,
                        laplace = 3L,
                        gamma   = 4L,
                        beta    = 5L)
  # data list for TMB
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
  # initial guess (your helper)
  theta_init <- initial_guess(Y, Xhat, homoskedastic)

  obj <- MakeADFun(
    data       = data_list,
    parameters = list(theta = theta_init),
    DLL        = "MLBC",
    silent     = TRUE
  )

  opt <- nlminb(
    start     = obj$par,
    objective = obj$fn,
    gradient  = obj$gr,
    control   = list(iter.max = 200)
  )

  H     <- obj$he(opt$par)
  d     <- ncol(Xhat)
  b_raw <- opt$par[1:d]
  V_raw <- solve(H)[1:d,1:d]

  # re-order if intercept
  if (intercept) {
    perm <- c(gen_idx, setdiff(seq_len(d), gen_idx))
    b    <- b_raw[perm]
    V    <- V_raw[perm, perm]
    names(b) <- colnames(Xhat)[perm]
  } else {
    b    <- b_raw
    V    <- V_raw
    names(b) <- colnames(Xhat)
  }

  out <- list(coef = b, cov = V)
  class(out) <- c("mlbc_fit", "mlbc_onestep")
  out
}

#' @rdname one_step
#' @method one_step formula
#' @importFrom stats model.frame model.response model.matrix
#' @export
one_step.formula <- function(Y,           # a one‐sided formula
                             Xhat = NULL, # ignored
                             homoskedastic = FALSE,
                             distribution  = c("normal","t","laplace","gamma","beta"),
                             nu            = 4,
                             gshape        = 2, gscale = 1,
                             ba            = 2, bb   = 2,
                             intercept     = TRUE,
                             gen_idx       = 1,
                             data          = parent.frame(),
                             ...) {

  # 1) build model frame + extract y
  mf <- stats::model.frame(Y, data = data)
  y  <- stats::model.response(mf)

  # 2) build the full design matrix Xm (includes its own intercept)
  Xm <- stats::model.matrix(Y, data = data)

  # 3) strip out ANY "(Intercept)" column
  if ("(Intercept)" %in% colnames(Xm)) {
    Xm <- Xm[, setdiff(colnames(Xm), "(Intercept)"), drop = FALSE]
  }

  # 4) locate which column is our ML‐variable by name
  rhs_terms <- attr(stats::terms(Y, data = data), "term.labels")
  ml_name   <- rhs_terms[1]  # first term on the right‐hand side
  gen_idx   <- match(ml_name, colnames(Xm))
  if (is.na(gen_idx)) {
    stop("Could not find ML‐variable '", ml_name, "' in the design matrix.")
  }

  # 5) hand off to default, which will
  #    - add its own intercept if intercept=TRUE
  #    - permute so ML‐column becomes col 1
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


