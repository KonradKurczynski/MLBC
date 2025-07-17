#' Job postings dataset
#'
#' A subset of data relating to job postings on the Lightcast platform
#' for demonstrating bias correction methods with ML-generated variables.
#'
#' @format ## `SD_data`
#' A data frame with 16315 rows and 6 columns:
#' \describe{
#'   \item{city_name}{Character. City of the job posting}
#'   \item{naics_2022_2}{Character. Type of business (NAICS industry classification)}
#'   \item{salary}{Numeric. Salary offered (response variable)}
#'   \item{wfh_wham}{Numeric. Binary label generated via ML, indicating whether
#'                   remote work is offered (subject to measurement error)}
#'   \item{soc_2021_2}{Character. Occupation code (SOC classification)}
#'   \item{employment_type_name}{Character. Employment type (part time/full time)}
#' }
#' @source Proprietary data from Lightcast job postings platform
#' @examples
#' \dontrun{
#' data(SD_data)
#' fit <- ols_bca(log(salary) ~ wfh_wham + soc_2021_2 + naics_2022_2,
#'                data = SD_data, fpr = 0.009, m = 1000)
#' }
"SD_data"
#' Topic model dataset
#'
#' Dataset containing topic model outputs for demonstrating bias correction
#' methods in topic model regressions using CEO diary data.
#'
#' @format A list with 8 components:
#' \describe{
#'   \item{covars}{Data frame (916 x 11): Control variables}
#'   \item{estimation_data}{Data frame (916 x 672): Contains outcome `ly` and word frequencies}
#'   \item{gamma_draws}{Data frame (2000 x 2): MCMC draws}
#'   \item{theta_est_full}{Data frame (916 x 2): Full sample topic proportions}
#'   \item{theta_est_samp}{Data frame (916 x 2): Subsample topic proportions}
#'   \item{beta_est_full}{Data frame (2 x 654): Full sample topic-word distributions}
#'   \item{beta_est_samp}{Data frame (2 x 654): Subsample topic-word distributions}
#'   \item{lda_data}{Data frame (916 x 2): LDA validation data}
#' }
#'
#' @source CEO diary data from Bandiera et al (2020), Journal of Political Economy
#'
#' @examples
#' data(topic_model_data)
#'
#' # Basic exploration
#' Y <- topic_model_data$estimation_data$ly
#' theta <- as.matrix(topic_model_data$theta_est_full)
#'
#' cat("Sample size:", length(Y), "\n")
#' cat("Mean log employment:", round(mean(Y), 2), "\n")
#' cat("Topic 1 mean:", round(mean(theta[, 1]), 3), "\n")
#'
#' @seealso \code{\link{ols_bca_topic}}, \code{\link{ols_bcm_topic}}
"topic_model_data"
