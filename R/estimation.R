#' One-step TMB estimator
#' @export
#' @importFrom TMB MakeADFun
one_step <- function(Y, Xhat, homoskedastic = FALSE, distribution = NULL) {
  data <- list(Y = as.numeric(Y), Xhat = as.matrix(Xhat),
               homoskedastic = as.integer(homoskedastic))
  theta_init <- initial_guess(Y, Xhat, homoskedastic)
  obj <- MakeADFun(
    data       = data,
    parameters = list(theta = theta_init),
    DLL        = "MLBC",
    silent     = TRUE
  )
  opt <- nlminb(start = obj$par, objective = obj$fn, gradient = obj$gr,
                control = list(iter.max = 200))
  H <- obj$he(opt$par)
  d <- ncol(Xhat)
  list(coef = opt$par[1:d], cov = solve(H)[1:d,1:d])
}
