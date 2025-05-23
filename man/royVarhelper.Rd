% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pcCovHelper.R
\name{royVarhelper}
\alias{royVarhelper}
\title{Helper Function for Calculating Asymptotic Covariance of Partial Correlations}
\usage{
royVarhelper(p, diags = FALSE, errors = FALSE)
}
\arguments{
\item{p}{number of variables in data set}

\item{diags}{Logical. (I think?) Only include indices for diagonal of covariance matrix (TRUE) or not (FALSE)}

\item{errors}{Logical. Create indices for correlations between errors (TRUE) or correlations between variables (FALSE)}
}
\value{
Matrix of indices for assisting in efficient calculation of asymptotic covariance matrix estimate using either royVar() or partialCov_cpp()
}
\description{
This function assists in the efficient calculation of an asymptotic covariance matrix estimate using either royVar() or partialCov_cpp()
}
\author{
Andrew DiLernia
}
