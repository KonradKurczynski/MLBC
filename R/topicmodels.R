#' Multiplicative bias-corrected OLS for topic models (BCM-Topic)
#'
#' Bias-corrected multiplicative estimator for topic model regression. This method
#' applies multiplicative bias correction to regressions that include topic proportions
#' as covariates, accounting for estimation uncertainty in the topic model.
#'
#' @inheritParams ols_bca_topic
#'
#' @return An object of class `mlbc_fit` and `mlbc_bcm_topic` with:
#'   - `coef`: bias-corrected coefficient estimates
#'   - `vcov`: adjusted variance-covariance matrix
#'
#' @examples
#' # Load topic model dataset
#' data(topic_model_data)
#'
#' # Extract components
#' Y <- topic_model_data$estimation_data$ly
#' Z <- as.matrix(topic_model_data$covars)
#' theta_full <- as.matrix(topic_model_data$theta_est_full)
#' beta_full <- as.matrix(topic_model_data$beta_est_full)
#' lda_data <- as.matrix(topic_model_data$lda_data)
#'
#' # Apply multiplicative bias correction
#' kappa <- mean(1.0 / lda_data[, 1]) * sqrt(nrow(lda_data))
#' S <- matrix(c(1.0, 0.0), nrow = 1)
#'
#' fit <- ols_bcm_topic(Y, Z, theta_full, S, beta_full, k = kappa)
#' summary(fit)
#' @export
ols_bcm_topic <- function(Y, Q = NULL, W, S, B, k,
                          data = parent.frame(), intercept = TRUE, ...) {
  UseMethod("ols_bcm_topic")
}

#' @rdname ols_bcm_topic
#' @method ols_bcm_topic default
#' @export
ols_bcm_topic.default <- function(Y, Q = NULL, W, S, B, k, data = parent.frame(), intercept = TRUE, ...){
  Y <- as.numeric(Y)
  if (!is.null(Q)) {
    Q <- as.matrix(Q)
  }
  W <- as.matrix(W)
  S <- as.matrix(S)
  B <- as.matrix(B)

  internal <- ols_bc_topic_internal(Y, Q, W, S, B, k, intercept)

  bb <- internal$b
  Gamma <- internal$Gamma
  V <- internal$V
  d <- internal$d
  topic_names <- internal$topic_names

  eigvals <- eigen(Gamma, only.values = TRUE)$values
  rho <- max(abs(eigvals))

  if (rho < 1) {

    b_raw <- solve(diag(d) - Gamma, bb)
    method_used <- "multiplicative"
  } else {

    b_raw <- (diag(d) + Gamma) %*% bb
    method_used <- "additive_fallback"
    warning("Spectral radius >= 1 (rho = ", round(rho, 4),
            "), using additive correction as fallback")
  }

  V_final <- V

  names(b_raw) <- topic_names
  colnames(V_final) <- rownames(V_final) <- topic_names

  if (!is.null(Q) || intercept) {
    out_coefs <- .reorder_topic_coefs(b_raw, V_final,
                                      ncol(S), has_intercept = intercept)
  } else {
    out_coefs <- list(coef = b_raw, vcov = V_final)
  }

  res <- list(
    coef = out_coefs$coef,
    vcov = out_coefs$vcov,
    method_used = method_used,
    spectral_radius = rho,
    nobs = length(Y)
  )
  class(res) <- c("mlbc_fit", "mlbc_bcm_topic")
  res
}

#' @rdname ols_bcm_topic
#' @method ols_bcm_topic formula
#' @importFrom stats model.frame model.response model.matrix terms
#' @export
ols_bcm_topic.formula <- function(Y, Q = NULL, W, S, B, k, data = parent.frame(), intercept = TRUE, ...){

  mf <- stats::model.frame(Y, data)
  y <- stats::model.response(mf)
  terms_obj <- stats::terms(mf)
  Xmat <- stats::model.matrix(terms_obj, mf)


  if ("(Intercept)" %in% colnames(Xmat)) {
    Xmat <- Xmat[, setdiff(colnames(Xmat), "(Intercept)"), drop = FALSE]
  }

  n_topics <- ncol(S)
  if (ncol(Xmat) > n_topics) {
    Q_from_formula <- Xmat[, (n_topics + 1):ncol(Xmat), drop = FALSE]
  } else {
    Q_from_formula <- NULL
  }

  ols_bcm_topic.default(y, Q = Q_from_formula, W = W, S = S, B = B, k = k,
                        intercept = intercept, ...)
}

#' Additive bias-corrected OLS for topic models (BCA-Topic)
#'
#' Bias-corrected additive estimator for topic model regression. This method
#' applies additive bias correction to regressions that include topic proportions
#' as covariates, accounting for estimation uncertainty in the topic model.
#'
#' @param Y numeric response vector, or a one-sided formula
#' @param Q numeric matrix of additional controls (if Y is numeric)
#' @param W numeric matrix of document-term frequencies
#' @param S numeric matrix of topic loadings
#' @param B numeric matrix of topic-word distributions
#' @param k numeric; bias correction parameter
#' @param data data frame (if Y is a formula)
#' @param intercept logical; if TRUE, includes an intercept term
#' @param ... additional arguments
#'
#' @return An object of class `mlbc_fit` and `mlbc_bca_topic` with:
#'   - `coef`: bias-corrected coefficient estimates
#'   - `vcov`: adjusted variance-covariance matrix
#'
#' @examples
#' # Load topic model dataset
#' data(topic_model_data)
#'
#' # Extract components
#' Y <- topic_model_data$estimation_data$ly
#' Z <- as.matrix(topic_model_data$covars)
#' theta_full <- as.matrix(topic_model_data$theta_est_full)
#' beta_full <- as.matrix(topic_model_data$beta_est_full)
#' lda_data <- as.matrix(topic_model_data$lda_data)
#'
#' # Apply additive bias correction
#' kappa <- mean(1.0 / lda_data[, 1]) * sqrt(nrow(lda_data))
#' S <- matrix(c(1.0, 0.0), nrow = 1)
#'
#' fit <- ols_bca_topic(Y, Z, theta_full, S, beta_full, k = kappa)
#' summary(fit)
#' @export
ols_bca_topic <- function(Y, Q = NULL, W, S, B, k,
                          data = parent.frame(), intercept = TRUE, ...) {
  UseMethod("ols_bca_topic")
}

#' @rdname ols_bca_topic
#' @method ols_bca_topic default
#' @export
ols_bca_topic.default <- function(Y, Q = NULL, W, S, B, k, data = parent.frame(), intercept = TRUE, ...){
  Y <- as.numeric(Y)
  if (!is.null(Q)) {
    Q <- as.matrix(Q)
  }
  W <- as.matrix(W)
  S <- as.matrix(S)
  B <- as.matrix(B)

  internal <- ols_bc_topic_internal(Y, Q, W, S, B, k, intercept)

  bb <- internal$b
  Gamma <- internal$Gamma
  V <- internal$V
  d <- internal$d
  topic_names <- internal$topic_names

  b_raw <- (diag(d) + Gamma) %*% bb

  V_final <- V

  names(b_raw) <- topic_names
  colnames(V_final) <- rownames(V_final) <- topic_names

  if (!is.null(Q) || intercept) {
    out_coefs <- .reorder_topic_coefs(b_raw, V_final,
                                      ncol(S), has_intercept = intercept)
  } else {
    out_coefs <- list(coef = b_raw, vcov = V_final)
  }

  res <- list(
    coef = out_coefs$coef,
    vcov = out_coefs$vcov,
    method_used = "additive",
    nobs = length(Y)
  )
  class(res) <- c("mlbc_fit", "mlbc_bca_topic")
  res
}

#' @rdname ols_bca_topic
#' @method ols_bca_topic formula
#' @importFrom stats model.frame model.response model.matrix terms
#' @export
ols_bca_topic.formula <- function(Y, Q = NULL, W, S, B, k, data = parent.frame(), intercept = TRUE, ...){
  mf <- stats::model.frame(Y, data)
  y <- stats::model.response(mf)
  terms_obj <- stats::terms(mf)
  Xmat <- stats::model.matrix(terms_obj, mf)

  if ("(Intercept)" %in% colnames(Xmat)) {
    Xmat <- Xmat[, setdiff(colnames(Xmat), "(Intercept)"), drop = FALSE]
  }

  n_topics <- ncol(S)
  if (ncol(Xmat) > n_topics) {
    Q_from_formula <- Xmat[, (n_topics + 1):ncol(Xmat), drop = FALSE]
  } else {
    Q_from_formula <- NULL
  }

  ols_bca_topic.default(y, Q = Q_from_formula, W = W, S = S, B = B, k = k,
                        intercept = intercept, ...)
}

#' @noRd
ols_bc_topic_internal <- function(Y, Q, W, S, B, k, intercept = TRUE) {

  theta <- W %*% t(S)

  if (!is.null(Q)) {
    Xhat <- cbind(theta, Q)
  } else {
    Xhat <- theta
  }

  if (intercept) {
    Xhat <- cbind(Xhat, Intercept = 1)
  }

  d <- ncol(Xhat)


  fit <- ols(Y, Xhat, se = TRUE, intercept = FALSE)
  b <- fit$coef
  V <- fit$vcov
  sXX <- fit$sXX


  n <- if (is.null(dim(Y))) length(Y) else nrow(Y)



  mW <- colMeans(W)
  BtB_inv <- solve(B %*% t(B))
  Bt_mW <- as.vector(t(B) %*% mW)
  M <- t(B) * Bt_mW

  Omega <- S %*% BtB_inv %*% B %*% M %*% BtB_inv %*% t(S) - (t(theta) %*% theta) / n

  A <- matrix(0, d, d)
  A[1:nrow(S), 1:nrow(S)] <- Omega

  Gamma <- (k / sqrt(n)) * solve(sXX, A)

  topic_names <- character(d)
  topic_names[1:ncol(theta)] <- paste0("topic", 1:ncol(theta))
  if (!is.null(Q)) {
    q_start <- ncol(theta) + 1
    q_end <- q_start + ncol(Q) - 1
    if (!is.null(colnames(Q))) {
      topic_names[q_start:q_end] <- colnames(Q)
    } else {
      topic_names[q_start:q_end] <- paste0("control", 1:ncol(Q))
    }
  }
  if (intercept) {
    topic_names[d] <- "Intercept"
  }

  list(
    b = b,
    Gamma = Gamma,
    V = V,
    d = d,
    topic_names = topic_names
  )
}

# -----------------------------------------------------------------------------
# Helper function for reordering topic model coefficients
# -----------------------------------------------------------------------------
#' Reorder topic model coefficients: topics first, controls next, intercept last
#' @param coef Named numeric vector of coefficients
#' @param vcov Variance-covariance matrix
#' @param n_topics Number of topic variables
#' @param has_intercept Whether an intercept is included
#' @return A list with elements `coef` and `vcov`, reordered
#' @keywords internal
#' @noRd
.reorder_topic_coefs <- function(coef, vcov, n_topics, has_intercept = TRUE) {
  nm <- names(coef)


  topic_idx <- 1:n_topics

  if (has_intercept) {
    int_idx <- which(nm %in% c("(Intercept)", "Intercept"))
    if (length(int_idx) == 0) {

      int_idx <- length(nm)
    }
  } else {
    int_idx <- integer(0)
  }

  control_idx <- setdiff(seq_along(nm), c(topic_idx, int_idx))

  perm <- c(topic_idx, control_idx, int_idx)
  perm <- perm[perm <= length(nm)]

  list(
    coef = coef[perm],
    vcov = vcov[perm, perm, drop = FALSE]
  )
}
