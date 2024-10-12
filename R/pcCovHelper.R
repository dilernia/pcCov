#' Simulate Data from a VAR Model
#'
#' This function simulates data from a mean 0 first-order vector auto-regressive (VAR) model.
#' @param nt Number of observations
#' @param coeffMat \eqn{p} x \eqn{p} matrix of coefficients
#' @param covMat \eqn{p} x \eqn{p} covariance matrix of residuals
#' @return \eqn{nt} x \eqn{p} matrix of the generated \eqn{p}-variate time series
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
#' # Plotting generated time series
#' matplot(myTS, type = 'l', lty = "solid", main = "Generated Multivariate Time Series",
#' xlab = "Time", ylab = "Value")
#'
#' @export
varSim <- function (nt, coeffMat, covMat) {
  wts <- MASS::mvrnorm(n = nt, mu = rep(0, ncol(covMat)), Sigma = covMat)
  ys <- matrix(0, nrow = nt, ncol = ncol(covMat))
  ys[1, ] <- wts[1, ]
  for (r in 2:nt) {
    ys[r, ] <- t(coeffMat %*% ys[r - 1, ]) + wts[r, ]
  }
  return(ys)
}

#' Partial Correlation Matrix
#'
#' This function uses the inverse covariance matrix to calculate the partial correlation matrix
#' @param icmat \eqn{p} x \eqn{p} inverse covariance matrix
#' @return \eqn{p} x \eqn{p} partial correlation matrix
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
#' # Calculate partial correlation matrix
#' invCov2part(solve(cov(myTS)))
#'
#' @export
invCov2part <- function(icmat) {
  p <- nrow(icmat)
  pmat <- matrix(0, ncol = p, nrow = p)
  for(i in 1:p) {
    for(j in 1:p) {
      pmat[i, j] <- -icmat[i, j] / sqrt(icmat[i, i]*icmat[j, j])
    }
  }
  return(pmat)
}

#' Helper Function for Calculating Asymptotic Covariance of Partial Correlations
#'
#' This function assists in the efficient calculation of an asymptotic covariance matrix estimate using either royVar() or partialCov_cpp()
#' @param p number of variables in data set
#' @param diags Logical. (I think?) Only include indices for diagonal of covariance matrix (TRUE) or not (FALSE)
#' @param errors Logical. Create indices for correlations between errors (TRUE) or correlations between variables (FALSE)
#' @return Matrix of indices for assisting in efficient calculation of asymptotic covariance matrix estimate using either royVar() or partialCov_cpp()
#'
#' @author
#' Andrew DiLernia
#'
#' @export
royVarhelper <- function(p, diags = FALSE, errors = FALSE) {

  q <- choose(p, 2)

  if(errors) {
    qs <- seq(2, 2*q, by = 2)
    bs  <- unlist(sapply(1:q, FUN = function(x) {qs[1:x]}))
    es <- unlist(sapply(1:q, FUN = function(x) {rep(qs[x], x)}))
    royCov <- cbind(a = bs - 1, b = bs, d = es - 1, e = es)
  } else {

    # Instantiating matrix for Roy covariances
    labs <- apply(expand.grid(1:p, 1:p)[as.vector(upper.tri(diag(p), diag = diags)),],
                  MARGIN = 1, FUN = paste, collapse = "_")
    rinds <- expand.grid(labs, labs)[as.vector(upper.tri(diag(q), diag = T)),]

    royCov <- as.matrix(data.frame(a = as.integer(gsub(rinds$Var1, pattern = "_.+$", replacement = "")),
                                   b = as.integer(gsub(rinds$Var1, pattern = ".+_", replacement = "")),
                                   d = as.integer(gsub(rinds$Var2, pattern = "_.+$", replacement = "")),
                                   e = as.integer(gsub(rinds$Var2, pattern = ".+_", replacement = ""))))
  }

  return(royCov)
}

# Function for calculating Wald test statistics
testStat <- function(cMat, royCov) {
  num12 <- cMat %*% (royCov[[1]]$beta - royCov[[2]]$beta)
  denom <- cMat %*% (royCov[[1]]$betaCov + royCov[[2]]$betaCov) %*% t(cMat)
  return(t(num12) %*% solve(denom, num12))
}

#' Roy's Asymptotic Covariance Matrix for Marginal Correlations
#'
#' This function calculates Roy (1989)'s asymptotic covariance matrix for marginal or partial correlations.
#' @param ts \eqn{nt} x \eqn{p} matrix of the observed \eqn{p}-variate time series
#' @param partial Logical. Calculate the asymptotic covariance matrix for the partial correlations (TRUE) or marginal correlations (FALSE)
#' @param bw Specified bandwidth (Optional). If not specified, optimal bandwidth is determined using the method described in Patton, Politis and White (2009).
#' @return \eqn{q} x \eqn{q} asymptotic covariance matrix where \eqn{q=}choose(\eqn{p}, 2).
#' For covariance estimator for partial correlations partial = TRUE, for covariance estimator for marginal correlations partial = FALSE.
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
#' # Asymptotic covariance matrix for partial correlations
#' royVar(myTS, partial = TRUE)
#'
#' @references
#' Roy, R. (1989). Asymptotic covariance structure of serial correlations in
#' multivariate time series, \emph{Biometrika}, 76(4), 824-827.
#'
#' Politis, D.N. and H. White (2004), Automatic block-length selection for the dependent bootstrap, \emph{Econometric Reviews}, 23(1), 53-70.
#'
#' @export
royVar <- function(ts, partial = FALSE, bw = NULL) {
  # Number of variables and observations
  p <- ncol(ts)
  N <- nrow(ts)
  q <- choose(p, 2)

  iMate <- royVarhelper(p = p, errors = partial)

  # Selecting optimal bandwidth if unspecified
  if(is.null(bw)) {
    bw <- ceiling(mean(np::b.star(ts)[, 1]))
  }

  if(partial == TRUE) {
    # Matrices of indices for unique partial correlations
    pairs <- subset(expand.grid(v1 = 1:p, v2 = 1:p), v1 < v2)

    # Roy variance estimator for marginal correlations of errors
    eps <- matrix(0, nrow = N, ncol = 2*q)
    for(r in 1:q) {
      i <- pairs[r, 1]
      j <- pairs[r, 2]
      eps[, (2*r-1):(2*r)] <- residuals(lm(ts[, c(i, j)] ~ 0 + ts[, -c(i, j)]))
    }

    pcCovHat <- royVar_cpp(iMat = iMate, tsData = eps, q = q, bw = bw)
  } else {
    pcCovHat <- royVar_cpp(iMat = iMate, tsData = ts, q = q, bw = bw)
  }
  return(pcCovHat)
}

#' @title Taylor Series Estimate of Covariance Matrix for Partial Correlations
#'
#' @description This function calculates a second-order Taylor Series estimate of the covariance matrix for partial correlations of a stationary Gaussian process.
#'
#' @param ts \eqn{nt} x \eqn{p} matrix of observed p-variate time series.
#' @param bw Specified bandwidth (Optional). If not specified, optimal bandwidth is determined using the method described in Patton, Politis and White (2009).
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
#' # Asymptotic covariance matrix for partial correlations
#' partialCov(ts = myTS)
#'
#' @references
#' Politis, D.N. and H. White (2004), Automatic block-length selection for the dependent bootstrap, \emph{Econometric Reviews}, 23(1), 53-70.
#'
#' @export
partialCov <- function(ts, bw = NULL) {
  p <- ncol(ts)
  q <- choose(p, 2)
  indMat <- pcCov::royVarhelper(p)
  indMate <- pcCov::royVarhelper(p, errors = T)
  iMatq <- unique(indMat[, 1:2])

  # Selecting optimal bandwidth if unspecified
  if(is.null(bw)) {
    bw <- ceiling(mean(np::b.star(ts)[, 1]))
  }
  return(pcCov::partialCov_cpp(ts = ts, bw = bw, iMatq = iMatq, iMate = indMate, q = q))
}

#' @title Taylor Series Estimate of Covariance Matrix for Partial Correlations
#'
#' @description This function calculates a second-order Taylor Series estimate of the covariance matrix for partial correlations of a stationary Gaussian process.
#'
#' @param ts \eqn{nt} x \eqn{p} matrix of observed p-variate time series.
#' @param bw Specified bandwidth (Optional). If not specified, optimal bandwidth is determined using the method described in Patton, Politis and White (2009).
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
#' # Asymptotic covariance matrix for partial correlations
#' partialCov(ts = myTS)
#'
#' @references
#' Politis, D.N. and H. White (2004), Automatic block-length selection for the dependent bootstrap, \emph{Econometric Reviews}, 23(1), 53-70.
#'
#' @export
partialCov2 <- function(ts, bw = NULL, method = "OLS", lambda = NULL) {
  p <- ncol(ts)
  q <- choose(p, 2)
  indMat <- pcCov::royVarhelper(p)
  indMate <- pcCov::royVarhelper(p, errors = T)
  iMatq <- unique(indMat[, 1:2])

  # Selecting optimal bandwidth if unspecified
  if(is.null(bw)) {
    bw <- ceiling(mean(np::b.star(ts)[, 1]))
  }

  if(method == "OLS") {
    pc_covariance <- pcCov::partialCov_cpp(ts = ts, bw = bw, iMatq = iMatq, iMate = indMate, q = q)
  } else if(method == "LASSO") {
    # Calculate residuals using LASSO penalized regression
    residual_matrix <- calculate_residuals_matrix(ts = ts, iMatq = iMatq, lambda = lambda)
    pc_covariance <- pcCov::partialCov_cpp2(ts = ts, bw = bw, iMatq = iMatq, iMate = indMate, q = q, resids = residual_matrix)
  }

  return(pc_covariance)
}
