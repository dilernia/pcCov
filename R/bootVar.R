#' Block-Bootstrap for Partial Correlations of a Multivariate Time Series
#'
#' This function calculates block-bootstrap covariance matrix estimates and confidence intervals for partial correlations of a multivariate time series.
#' @param ts \eqn{nt} x \eqn{p} matrix of the observed \eqn{p}-variate time series
#' @param winLength Specified window length for block-bootstrap (Optional). If not specified, block-length is determined using the method described in Patton, Politis and White (2009).
#' @param nboots Number of boot-strap samples
#'
#' @return A list of length 2 containing:
#' \enumerate{
#' \item \eqn{q} x \eqn{q} estimated covariance matrix of the partial correlations where \eqn{q=}choose(\eqn{p}, 2)
#' \item \eqn{q} x \eqn{2} matrix containing the lower (first column) and upper (second column) bounds of the 95\% bootstrap confidence intervals.
#' }
#'
#' @author
#' Andrew DiLernia
#'
#' @examples
#' # Generate multivariate time series with 5 variables from a
#' # first-order VAR model with 50 time points
#' set.seed(1994)
#' myTS <- varSim(nt = 50, coeffMat = diag(0.50, 5), covMat = diag(1, 5))
#'
#' # Estimated covariance matrix for partial correlations and 95% bootstrap intervals
#' bootRes <- bootVar(myTS)
#'
#' @references
#' Politis, D.N. and H. White (2004), Automatic block-length selection for the dependent bootstrap, \emph{Econometric Reviews}, 23(1), 53-70.
#'
#' @export
bootVar <- function(ts, winLength = NULL, nboots = 1000) {
  p <- ncol(ts)
  q <- choose(p, 2)
  N <- nrow(ts)
  if(is.null(winLength)) {
    winLength <- ceiling(mean(np::b.star(ts)[, 1]))
  }
  nBlocks <- round(N / (winLength))
  inds <- seq(N - winLength + 1)

  bootSamps <- t(simplify2array(lapply(1:nboots, FUN = function(x) {
    bootsamp <- scale(do.call('rbind', lapply(1:nBlocks, FUN = function(b) {
      istart <- sample(inds, 1)
      ts[(1:N)[istart:(istart + winLength -1)], ]})))
    pcs <- corrMat_cpp(bootsamp, partial = TRUE)[upper.tri(diag(p), diag = FALSE)]
    return(pcs)})))

  bsPcCov <- cov(bootSamps)
  bsCis <- t(apply(bootSamps, FUN = function(pcs){quantile(pcs, probs = c(0.025, 0.975), na.rm = TRUE)},
                   MARGIN = 2))

  return(list(bsPcCov, bsCis))
}
