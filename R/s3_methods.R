#' @export
print.mlbc_fit <- function(x, ...) {
  cat("<MLBC fit - class:", paste(class(x), collapse = "/"), ">\n")
  cat("Coefficients:\n")
  print(x$coef)
  invisible(x)
}

#' @export
summary.mlbc_fit <- function(object, level = 0.95, ...) {
  coefs <- object$coef
  vcov_mat <- object$vcov

  if (is.null(vcov_mat)) {
    stop("Variance-covariance matrix not available")
  }

  ses <- sqrt(diag(vcov_mat))
  p <- length(coefs)

  alpha <- 1 - level
  z_crit <- qnorm(1 - alpha/2)

  t_stats <- coefs / ses
  p_values <- 2 * pnorm(-abs(t_stats))

  ci_lower <- coefs - z_crit * ses
  ci_upper <- coefs + z_crit * ses

  sig_stars <- ifelse(p_values < 0.001, "***",
                      ifelse(p_values < 0.01, "**",
                             ifelse(p_values < 0.05, "*",
                                    ifelse(p_values < 0.1, ".", ""))))

  orig_names <- names(coefs)
  if (is.null(orig_names) || any(orig_names == "")) {
    orig_names <- paste0("X", seq_len(p))
  }

  if ("(Intercept)" %in% orig_names) {
    intercept_idx <- which(orig_names == "(Intercept)")
  } else if ("Intercept" %in% orig_names) {
    intercept_idx <- which(orig_names == "Intercept")
  } else {
    intercept_idx <- p
  }
  slope_idx <- setdiff(seq_len(p), intercept_idx)

  ord <- c(intercept_idx, slope_idx)
  beta_labels <- paste0("Beta_", seq_len(p) - 1)

  results_table <- data.frame(
    Estimate = coefs[ord],
    Std.Error = ses[ord],
    z.value = t_stats[ord],
    `Pr(>|z|)` = p_values[ord],
    `CI.Lower` = ci_lower[ord],
    `CI.Upper` = ci_upper[ord],
    Signif = sig_stars[ord],
    row.names = beta_labels,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  rhs_terms <- paste0(
    beta_labels[-1],
    " * ",
    orig_names[ord[-1]]
  )
  formula_str <- paste(
    "Y ~",
    beta_labels[1],
    if (length(rhs_terms) > 0) paste("+", paste(rhs_terms, collapse = " + ")) else ""
  )

  attr(results_table, "call") <- formula_str
  attr(results_table, "confidence_level") <- level
  attr(results_table, "nobs") <- if(!is.null(object$nobs)) object$nobs else NA
  attr(results_table, "loglik") <- if(!is.null(object$loglik)) object$loglik else NA
  attr(results_table, "convergence") <- if(!is.null(object$convergence)) object$convergence else NA
  attr(results_table, "vcov") <- vcov_mat[ord, ord, drop = FALSE]


  class(results_table) <- c("mlbc_summary", "data.frame")
  results_table
}

#' @export
print.mlbc_summary <- function(x, digits = 4, signif.stars = TRUE, ...) {
  cat("\n")
  cat("MLBC Model Summary\n")
  cat("==================\n\n")

  call_attr <- attr(x, "call")
  if (!is.null(call_attr)) {
    cat("Formula: ", call_attr, "\n\n")
  }

  nobs_attr <- attr(x, "nobs")
  if (!is.na(nobs_attr)) {
    cat("Number of observations:", nobs_attr, "\n")
  }

  loglik_attr <- attr(x, "loglik")
  if (!is.na(loglik_attr)) {
    cat("Log-likelihood:", round(loglik_attr, digits), "\n")
  }

  convergence_attr <- attr(x, "convergence")
  if (!is.na(convergence_attr) && convergence_attr != 0) {
    cat("Convergence code:", convergence_attr, "(check convergence!)\n")
  }

  cat("\nCoefficients:\n")

  display_table <- as.data.frame(x)
  class(display_table) <- "data.frame"

  numeric_cols <- sapply(display_table, is.numeric)
  for (col_name in names(display_table)[numeric_cols]) {
    display_table[[col_name]] <- round(display_table[[col_name]], digits)
  }


  if ("Pr(>|z|)" %in% names(display_table)) {
    p_vals <- x[["Pr(>|z|)"]]  # Use original values from x
    display_table[["Pr(>|z|)"]] <- ifelse(p_vals < 2e-16, "< 2e-16",
                                          ifelse(p_vals < 1e-4, sprintf("%.2e", p_vals),
                                                 sprintf("%.4f", p_vals)))
  }


  if ("CI.Lower" %in% names(display_table) && "CI.Upper" %in% names(display_table)) {
    ci_level <- attr(x, "confidence_level")
    if (is.null(ci_level)) ci_level <- 0.95
    ci_pct <- paste0(round(ci_level * 100), "% CI")

    ci_lower_orig <- x$CI.Lower  # Use original values
    ci_upper_orig <- x$CI.Upper

    display_table[[ci_pct]] <- paste0("[",
                                      sprintf("%.4f", ci_lower_orig),
                                      ", ",
                                      sprintf("%.4f", ci_upper_orig),
                                      "]")

    display_table$CI.Lower <- NULL
    display_table$CI.Upper <- NULL
  }


  cat("\n")
  print.data.frame(display_table, quote = FALSE, right = TRUE)

  if (signif.stars) {
    cat("---\n")
    cat("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
  }

  cat("\n")
  invisible(x)
}

# Method-specific summary methods
#' @export
summary.mlbc_onestep <- function(object, level = 0.95, ...) {
  summary_obj <- summary.mlbc_fit(object, level = level, ...)
  attr(summary_obj, "method") <- "One-step TMB estimation"
  attr(summary_obj, "distribution") <- attr(object, "distribution")
  class(summary_obj) <- c("mlbc_summary_onestep", class(summary_obj))
  summary_obj
}

#' @export
print.mlbc_summary_onestep <- function(x, ...) {
  cat("\n")
  cat("MLBC One-Step TMB Model Summary\n")
  cat("===============================\n")

  method_attr <- attr(x, "method")
  if (!is.null(method_attr)) {
    cat("Method:", method_attr, "\n")
  }

  distribution_attr <- attr(x, "distribution")
  if (!is.null(distribution_attr)) {
    cat("Distribution:", distribution_attr, "\n")
  }

  NextMethod("print")
}

#' @export
summary.mlbc_bca <- function(object, level = 0.95, ...) {
  summary_obj <- summary.mlbc_fit(object, level = level, ...)
  attr(summary_obj, "method") <- "Additive bias-corrected OLS (BCA)"
  attr(summary_obj, "bias_correction") <- "Additive"
  class(summary_obj) <- c("mlbc_summary_bca", class(summary_obj))
  summary_obj
}

#' @export
summary.mlbc_bcm <- function(object, level = 0.95, ...) {
  summary_obj <- summary.mlbc_fit(object, level = level, ...)
  attr(summary_obj, "method") <- "Multiplicative bias-corrected OLS (BCM)"
  attr(summary_obj, "bias_correction") <- "Multiplicative"
  class(summary_obj) <- c("mlbc_summary_bcm", class(summary_obj))
  summary_obj
}

#' @export
print.mlbc_summary_bca <- function(x, ...) {
  cat("\n")
  cat("MLBC Additive Bias-Corrected Model Summary\n")
  cat("==========================================\n")
  cat("Method: Additive bias-corrected OLS (BCA)\n")
  NextMethod("print")
}

#' @export
print.mlbc_summary_bcm <- function(x, ...) {
  cat("\n")
  cat("MLBC Multiplicative Bias-Corrected Model Summary\n")
  cat("================================================\n")
  cat("Method: Multiplicative bias-corrected OLS (BCM)\n")
  NextMethod("print")
}

# Other methods remain the same
#' @export
coef.mlbc_fit <- function(object, ...) {
  object$coef
}

#' @export
vcov.mlbc_fit <- function(object, ...) {
  object$vcov
}

#' @export
confint.mlbc_fit <- function(object, parm, level = 0.95, ...) {
  if (is.null(object$vcov)) {
    stop("Variance-covariance matrix not available")
  }

  coefs <- object$coef
  ses <- sqrt(diag(object$vcov))

  if (!missing(parm)) {
    if (is.character(parm)) {
      parm_idx <- match(parm, names(coefs))
      if (any(is.na(parm_idx))) {
        stop("Some parameters not found: ", paste(parm[is.na(parm_idx)], collapse = ", "))
      }
    } else {
      parm_idx <- parm
    }
    coefs <- coefs[parm_idx]
    ses <- ses[parm_idx]
  }

  alpha <- 1 - level
  z_crit <- qnorm(1 - alpha/2)

  ci_lower <- coefs - z_crit * ses
  ci_upper <- coefs + z_crit * ses

  ci_matrix <- cbind(ci_lower, ci_upper)
  colnames(ci_matrix) <- paste0(c(alpha/2, 1 - alpha/2) * 100, "%")
  rownames(ci_matrix) <- names(coefs)

  ci_matrix
}

# Topic model print methods remain the same
#' @export
print.mlbc_bcm_topic <- function(x, ...) {
  cat("<MLBC Topic Model BCM fit \n")
  cat("Method used:", x$method_used, "\n")
  if (!is.null(x$spectral_radius)) {
    cat("Spectral radius:", round(x$spectral_radius, 4), "\n")
  }
  cat("Coefficients:\n")
  print(x$coef)
  invisible(x)
}

#' @export
print.mlbc_bca_topic <- function(x, ...) {
  cat("<MLBC Topic Model BCA fit")
  cat("Method used:", x$method_used, "\n")
  cat("Coefficients:\n")
  print(x$coef)
  invisible(x)
}

#' @export
summary.mlbc_bcm_topic <- function(object, level = 0.95, ...) {
  summary_obj <- summary.mlbc_fit(object, level = level, ...)
  attr(summary_obj, "method") <- "Multiplicative bias-corrected OLS for topic models (BCM-Topic)"
  attr(summary_obj, "bias_correction") <- "Multiplicative (topic-specific)"
  attr(summary_obj, "method_used") <- object$method_used
  attr(summary_obj, "spectral_radius") <- object$spectral_radius
  class(summary_obj) <- c("mlbc_summary_bcm_topic", class(summary_obj))
  summary_obj
}

#' @export
summary.mlbc_bca_topic <- function(object, level = 0.95, ...) {
  summary_obj <- summary.mlbc_fit(object, level = level, ...)
  attr(summary_obj, "method") <- "Additive bias-corrected OLS for topic models (BCA-Topic)"
  attr(summary_obj, "bias_correction") <- "Additive (topic-specific)"
  class(summary_obj) <- c("mlbc_summary_bca_topic", class(summary_obj))
  summary_obj
}

#' @export
print.mlbc_summary_bcm_topic <- function(x, ...) {
  cat("\n")
  cat("MLBC Topic Model BCM Summary\n")
  cat("============================================\n")
  cat("Method: Multiplicative bias-corrected OLS for topic models\n")

  method_used_attr <- attr(x, "method_used")
  if (!is.null(method_used_attr)) {
    cat("Correction applied:", method_used_attr, "\n")
  }

  spectral_radius_attr <- attr(x, "spectral_radius")
  if (!is.null(spectral_radius_attr)) {
    cat("Spectral radius:", round(spectral_radius_attr, 4), "\n")
  }

  NextMethod("print")
}

#' @export
print.mlbc_summary_bca_topic <- function(x, ...) {
  cat("\n")
  cat("MLBC Topic Model BCA Summary\n")
  cat("============================================\n")
  cat("Method: Additive bias-corrected OLS for topic models\n")
  NextMethod("print")
}
