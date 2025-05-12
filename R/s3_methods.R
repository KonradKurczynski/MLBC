#' @export
print.mlbc_fit <- function(x, ...) {
  cat("<MLBC fit - class:", paste(class(x), collapse = "/"), ">\n")
  cat("Coefficients:\n")
  print(x$coef)
  invisible(x)
}

#' @export
summary.mlbc_fit <- function(object, ...) {
  coefs <- object$coef
  ses   <- sqrt(diag(object$vcov))
  p     <- length(coefs)


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


  beta_labels <- paste0("Beta_", seq_len(p) - 1)  # 0,1,2,...


  rhs_terms <- paste0(
    beta_labels[-1],
    " * ",
    orig_names[ord[-1]]
  )
  formula_str <- paste(
    "Y ~",
    beta_labels[1],
    if (length(rhs_terms) > 0) paste("+", paste(rhs_terms, collapse = " +")) else ""
  )
  cat(formula_str, "\n\n")


  df <- data.frame(
    Estimate  = coefs[ord],
    Std.Error = ses[ord],
    row.names = beta_labels,
    check.names = FALSE
  )
  class(df) <- c("mlbc_summary", class(df))
  df
}

#' @export
print.mlbc_summary <- function(x, ...) {
  cat("<MLBC model summary>\n")
  NextMethod("print")
}

#' @export
print.mlbc_summary <- function(x, ...) {
  cat("<MLBC model summary>\n")
  NextMethod("print")
}

#' @export
print.mlbc_summary <- function(x, ...) {
  cat("<MLBC model summary>\n")
  NextMethod("print")
}

#' @export
print.mlbc_summary <- function(x, ...) {
  cat("<MLBC summary>\n")
  print.data.frame(x)
  invisible(x)
}

#' @export
coef.mlbc_fit <- function(object, ...) {
  object$coef
}

#' @export
vcov.mlbc_fit <- function(object, ...) {
  object$vcov
}

#' @export
print.mlbc_onestep <- function(x, ...) {
  cat("<MLBC one-step TMB fit>\n")
  NextMethod("print")
}

#' @export
summary.mlbc_onestep <- function(object, ...) {
  tbl <- NextMethod("summary")
  class(tbl) <- c("mlbc_summary_onestep", class(tbl))
  tbl
}

#' @export
print.mlbc_summary_onestep <- function(x, ...) {
  cat("<MLBC one-step TMB summary>\n")
  NextMethod("print")
}
