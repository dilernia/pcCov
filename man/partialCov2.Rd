% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pcCovHelper.R
\name{partialCov2}
\alias{partialCov2}
\title{Taylor Series Estimate of Covariance Matrix for Partial Correlations}
\usage{
partialCov2(ts, bw = NULL, method = "OLS", lambda = NULL)
}
\arguments{
\item{ts}{\eqn{nt} x \eqn{p} matrix of observed p-variate time series.}

\item{bw}{Specified bandwidth (Optional). If not specified, optimal bandwidth is determined using the method described in Patton, Politis and White (2009).}
}
\description{
This function calculates a second-order Taylor Series estimate of the covariance matrix for partial correlations of a stationary Gaussian process.
}
\examples{
# Generate multivariate time series with 5 variables from a
# first-order VAR model with 50 time points
set.seed(1994)
myTS <- varSim(nt = 50, coeffMat = diag(0.50, 5), covMat = diag(1, 5))

# Asymptotic covariance matrix for partial correlations
partialCov(ts = myTS)

}
\references{
Politis, D.N. and H. White (2004), Automatic block-length selection for the dependent bootstrap, \emph{Econometric Reviews}, 23(1), 53-70.
}
\author{
Andrew DiLernia
}
