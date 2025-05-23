% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pcCovHelper.R
\name{partial_corr_asymptotic_cov}
\alias{partial_corr_asymptotic_cov}
\title{Taylor Series Estimate of Covariance Matrix for Partial Correlations}
\usage{
partial_corr_asymptotic_cov(
  mvts,
  bandwidth = NULL,
  structure = "unstructured",
  residuals = NULL,
  residual_pairs = NULL,
  correlation_pairs = NULL,
  correlation_indices = NULL
)
}
\arguments{
\item{mvts}{\eqn{nt} x \eqn{p} matrix of observed \eqn{p}-variate time series.}

\item{bandwidth}{Optional nonnegative bandwidth parameter. If not specified, optimal bandwidth is determined using the method described in Patton, Politis and White (2009).}

\item{structure}{Optional covariance structure, indicating whether to use a covariance estimator in which every entry is estimated ("unstructured"), covariances between correlations with disjoint pairs of variables are set to 0 while all others are estimated ("intersecting-pairs"), or off-diagonal entries are set to 0 and diagonal entries are estimated ("diagonal").}

\item{residuals}{Optional \eqn{nt} x \eqn{2q} matrix of empirical residuals (optional). If not specified, residuals are obtained using ordinary least squares.}

\item{residual_pairs}{Optional (choose(\eqn{q}, 2) + \eqn{q}) x 4 matrix of indices for residual pairs equal to royVarhelper(p, errors = TRUE).}

\item{correlation_pairs}{Optional (choose(\eqn{q}, 2) + \eqn{q}) x 4 matrix of indices for partial correlation pairs equal to royVarhelper(p, errors = FALSE).}

\item{correlation_indices}{Optional matrix of indices for partial correlations equal to unique(royVarhelper(p)[, 1:2]).}
}
\value{
\eqn{q} x \eqn{q} covariance matrix
}
\description{
This function calculates the second-order Taylor Series estimate of the covariance matrix for partial correlations of a weakly stationary multivariate time series proposed by DiLernia, Fiecas, and Zhang (2024).
}
\examples{
# Generate multivariate time series with 5 variables from a
# first-order VAR model with 50 time points
set.seed(1994)
myTS <- varSim(nt = 50, coeffMat = diag(0.50, 5), covMat = diag(1, 5))

# Asymptotic covariance matrix for partial correlations
partial_corr_asymptotic_cov(mvts = myTS, bandwidth = 4, structure = "unstructured")

}
\references{
A S DiLernia, M Fiecas, L Zhang.
Inference for partial correlations of a multivariate Gaussian time series.
\emph{Biometrika}, Volume 111, Issue 4, December 2024, Pages 1437–1444.
https://doi.org/10.1093/biomet/asae012

Politis, D.N., and H. White.
Automatic block-length selection for the dependent bootstrap.
\emph{Econometric Reviews}, 23(1), 53-70.
}
\author{
Andrew DiLernia
}
