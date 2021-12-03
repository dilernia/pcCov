#' Multi-Subject Block-Bootstrap for Multivariate Time Series
#'
#' This function obtains block-bootstrap samples for multi-subject time series data.
#' @param mvts List of \eqn{K} number of \eqn{nt} x \eqn{p} matrices of the observed \eqn{p}-variate time series.
#' @param winLength Specified window length for block-bootstrap (Optional). If not specified, block-length is determined using the method described in Patton, Politis and White (2009).
#' @param nboots Number of bootstrap samples.
#'
#' @return A list of \eqn{K} arrays containing block-bootstrap samples.
#'
#' @author
#' Andrew DiLernia
#'
#' @examples
#' # Generate multivariate time series for 10 subjects with 5 variables from a
#' # first-order VAR model with 100 time points
#' set.seed(1994)
#' multiSubjTS <- lapply(1:10, FUN = function(subj) {pcCov::varSim(nt = 100, coeffMat = diag(0.50, 5), covMat = diag(1, 5))})
#'
#' # Obtain 1000 block-bootstrap samples for each subject
#' bootRes <- multiBlockBoot(mvts = multiSubjTS, nboots = 1000)
#'
#' @references
#' Politis, D.N. and H. White (2004), Automatic block-length selection for the dependent bootstrap, \emph{Econometric Reviews}, 23(1), 53-70.
#'
#' @export
multiBlockBoot <- function(mvts, winLength = NULL, nboots = 500) {
  p <- ncol(mvts[[1]])
  Ns <- sapply(mvts, FUN = nrow)
  K <- length(mvts)

  if (is.null(winLength)) {
    winLength <- ceiling(mean(sapply(mvts, FUN = function(ts) {
        apply(ts, MARGIN = 2, FUN = function(datf) {np::b.star(datf)[, 1]})
      })))
  }

  nBlocks <- sapply(Ns, FUN = function(n){round(n/(winLength))})
  inds <- lapply(Ns, FUN = function(n){seq(n - winLength + 1)})

  istarts <- lapply(1:length(inds), FUN = function(k){
      sample(inds[[k]], size = nboots*nBlocks[k], replace = TRUE)})

  bootSamps <- lapply(1:K, FUN = function(k) {
      bootsamp <- sapply(1:nboots, simplify = "array", FUN = function(x) {
        sapply(1:nBlocks[k], simplify = "array", FUN = function(b) {
          istart <- istarts[[k]][b + nBlocks[k]*(x-1)]
          return(mvts[[k]][(1:Ns[[k]])[istart:(istart + winLength - 1)], ])})
      })
      return(array(aperm(bootsamp, c(1, 3, 2, 4)), dim = c(winLength*nBlocks[k], p, nboots)))
    })

  return(bootSamps)
}
