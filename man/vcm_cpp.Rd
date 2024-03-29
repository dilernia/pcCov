% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{vcm_cpp}
\alias{vcm_cpp}
\title{Variance Components Model}
\usage{
vcm_cpp(
  rs,
  sigmas,
  sigEigs,
  delta = 0.001,
  maxIters = 100L,
  sig0 = 0.1,
  smallRet = FALSE
)
}
\arguments{
\item{rs}{Column vector containing q x K unique marginal or partial correlations.}

\item{sigmas}{3D array of K estimated q x q covariance matrices for correlations.}

\item{sigEigs}{List of K matrices containing eigen decomposition matrices for covariance matrices contained in sigmas.}

\item{delta}{Threshold for algorithm}

\item{maxIters}{Maximum number of iterations for algorithm}

\item{sig0}{Initial value for sigma parameter}

\item{smallRet}{Logical value, whether or not to return smaller set of results}
}
\value{
List of length 4 containing beta (q x 1), betaCov (q x q), sigma (qK x qK), and psi (qK x qK) estimates. If smallRet = true, then only returns beta and betaCov.
}
\description{
This function implements the variance components model proposed by Fiecas et al. (2017).
}
\references{
Fiecas, M., Cribben, I., Bahktiari, R., and Cummine, J. (2017). A variance components model for statistical inference on functional connectivity networks. NeuroImage (Orlando, Fla.), 149, 256-266.
}
\author{
Andrew DiLernia
}
