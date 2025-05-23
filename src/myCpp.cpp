#include <cstdlib>
#include <iostream>
#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <wishart.h>
#include <mvnorm.h>
#include <Rmath.h>
#include <cmath>

// [[Rcpp::depends(RcppArmadillo, RcppEigen)]]

using namespace arma;
using namespace Eigen;
using namespace Rcpp;

//' Adjust matrix to be symmetric
//'
//' @param myMat A matrix.
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat mkSymm_cpp(arma::mat myMat) {
  arma::mat symmMat = (myMat + myMat.t()) / 2;
  return symmMat;
}

//' Calculate inverse of matrix
//'
//' @param myMat A matrix.
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat inv_cpp(arma::mat myMat) {
  return myMat.i();
}

//' Convert Eigen matrix to Arma matrix
//'
//' @param eigen_A A matrix.
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat eigen2arma(Eigen::MatrixXd eigen_A) {
  arma::mat arma_B = arma::mat(eigen_A.data(), eigen_A.rows(), eigen_A.cols(),
                               true,   // changed from false to true.
                               false);
  return arma_B;
}

//' Convert Arma matrix to Eigen matrix
//'
//' @param arma_A A matrix.
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
Eigen::MatrixXd arma2eigen(arma::mat arma_A) {

  Eigen::MatrixXd eigen_B = Eigen::Map<Eigen::MatrixXd>(arma_A.memptr(),
                                                        arma_A.n_rows,
                                                        arma_A.n_cols);

  return eigen_B;
}

//' Multiply two matrices
//'
//' @param A A matrix.
//' @param B A matrix.
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat eigenMult2(arma::mat A, arma::mat B) {
  arma::mat C = A * B;

  return C;
}

//' Multiply three matrices
//'
//' @param A A matrix.
//' @param B A matrix.
//' @param C A matrix.
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat eigenMult3(arma::mat A, arma::mat B, arma::mat C){

  arma::mat D = A * B * C;

  return(D);
}

//' Multiply four matrices
//'
//' @param A A matrix.
//' @param B A matrix.
//' @param C A matrix.
//' @param D A matrix.
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat eigenMult4(arma::mat A, arma::mat B, arma::mat C, arma::mat D){

  arma::mat E = A * B * C * D;

  return(E);
}

//' Convert inverse-covariance matrix to partial correlation matrix
//'
//' @param icmat An inverse-covariance matrix.
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat invCov2part_cpp(arma::mat icmat) {
  int p = icmat.n_rows;
  arma::mat pmat(p, p);
  for(int i = 0; i < p; i++) {
    for(int j = 0; j < p; j++) {
      pmat(i, j) = -icmat(i, j) / sqrt(icmat(i, i)*icmat(j, j));
    }
  }
  return(pmat);
}

//' Convert covariance matrix to correlation matrix
//'
//' @param cmat A covariance matrix.
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat cov2corr_cpp(arma::mat cmat) {
  int p = cmat.n_rows;
  arma::mat corrMat(p, p);
  for(int i = 0; i < p; i++) {
    for(int j = 0; j < p; j++) {
      corrMat(i, j) = cmat(i, j) / sqrt(cmat(i, i)*cmat(j, j));
    }
  }
  return(corrMat);
}

//' Calculate marginal or partial correlation matrix
//'
//' @param tsData An n x p data matrix.
//' @param partial Logical. Whether to calculate partial (TRUE) or marginal (FALSE) correlation matrix
//'
//' @return p x p correlation matrix
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat corrMat_cpp(arma::mat tsData, bool partial = true) {
  arma::mat cmat = arma::cov(tsData);
  if(partial) {
    cmat = invCov2part_cpp(arma::inv(cmat));
    return(cmat);
  } else {
    cmat = cov2corr_cpp(cmat);
    return(cmat);
  }
}

//' Function for Hann window taper
//'
//' @param u Vector of indices for window
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::vec cosTaper_cpp(arma::vec u) {

  // Calculate the sine values
  arma::vec sinValues = sin(M_PI * (u - u(0)) / u.n_elem);

  // Perform the sine and square operations
  arma::vec sqrtRet = square(sinValues);

  return sqrtRet;
}

//' Function for exponential window taper
//'
//' @param u Vector of indices for window
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
NumericVector expTaper_cpp(IntegerVector u){
  IntegerVector uNew = u - min(u);
  NumericVector ret = exp((-abs(as<NumericVector>(uNew) - u.size() / 2)) / (u.size() / 2 * 8.69 / 60));
  return(ret);
}

//' Calculate cross-covariance of two vectors
//'
//' @param u Vector of indices for window
//' @param ts1 First time series vector
//' @param ts2 Second time series vector
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
double crossCov_cpp(int u, arma::vec ts1, arma::vec ts2) {
  int N = ts1.n_elem;
  if(u < 0){
    arma::vec x1 = ts1(span(-u, N-1));
    arma::vec x2 = ts2(span(0, N+u-1));
    return(sum(x1 % x2) / N);
  } else{
    arma::vec x1 = ts1(span(0, N-u-1));
    arma::vec x2 = ts2(span(u, N-1));
    return(sum(x1 % x2) / (N - 1));
  }
}

//' Calculate cross-covariance of two vectors
//'
//' @param u Vector of indices for window
//' @param ts1 First time series vector
//' @param ts2 Second time series vector
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat crossCov2_cpp(int u, arma::vec ts1, arma::vec ts2) {
  int N = ts1.n_elem;
  arma::mat ret;
  if(u < 0){
    arma::vec x1 = ts1(span(-u, N-1));
    arma::vec x2 = ts2(span(0, N+u-1));
    ret = cov(x1, x2);
    return(ret);
  } else{
    arma::vec x1 = ts1(span(0, N-u-1));
    arma::vec x2 = ts2(span(u, N-1));
    ret = cov(x1, x2);
    return(ret);
  }
}

//' Calculate taper for two vectors
//'
//' @param ts1 First time series vector
//' @param ts2 Second time series vector
//' @param banw Bandwidth parameter
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat taperCov_cpp(arma::vec ts1, arma::vec ts2, int banw) {
  int N = ts1.n_elem;
  arma::uvec non0 = regspace<uvec>(0, banw);
  arma::vec hu2s = cosTaper_cpp(regspace<vec>(-banw, banw)).elem(regspace<uvec>(banw - 1, 2 * banw - 1));
  hu2s(0) = 1;
  int uLength = banw + 1;

  // Calculating cross-covariance for each lag
  arma::vec ccs(uLength);
  ccs(0) = crossCov_cpp(0, ts1, ts2);
  for(int i = 1; i < uLength; i++) {
    ccs[i] = crossCov_cpp(non0(i), ts1, ts2);
  }

  // Constructing tapered estimate for residual covariance matrix
  ccs = ccs % hu2s;

  arma::mat ccst = ccs.t();

  arma::mat sigma(N, N, fill::zeros);

  for(int i = 0; i < N - banw; i++) {
    sigma(i, span(i, i + banw)) = ccst;
  }

  int counter = 0;

  for(int i = N - banw; i < N; i++) {
    counter++;
    sigma(i, span(i, i + banw - counter)) = ccst(0, span(0, banw - counter));
  }

  sigma.diag() = sigma.diag() / 2;

  return(sigma + sigma.t());
}

//' Calculate taper for two vectors
//'
//' @param ts1 First time series vector
//' @param ts2 Second time series vector
//' @param banw Bandwidth parameter
//' @param hu2s Weights from selected taper function
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat taperCovSub_cpp(arma::vec ts1, arma::vec ts2, int banw, arma::vec hu2s) {
  arma::uvec non0 = regspace<uvec>(0, banw);
  int uLength = banw + 1;

  // Calculating cross-covariance for each lag
  arma::vec ccs(uLength);
  ccs(0) = crossCov_cpp(0, ts1, ts2);
  for(int i = 1; i < uLength; i++) {
    ccs[i] = crossCov_cpp(non0(i), ts1, ts2);
  }

  // Constructing tapered estimate for residual covariance matrix
  ccs = ccs % hu2s;
  arma::mat ccst = ccs.t();
  int banw2 = pow(banw + 1, 2);
  arma::mat sigma(banw2, banw2, fill::zeros);

  for(int i = 0; i < banw2 - banw; i++) {
    sigma(i, span(i, i + banw)) = ccst;
  }

  int counter = 0;

  for(int i = banw2 - banw; i < banw2; i++) {
    counter++;
    sigma(i, span(i, i + banw - counter)) = ccst(0, span(0, banw - counter));
  }

  sigma.diag() = sigma.diag() / 2;

  return(sigma + sigma.t());
}

//' Calculate upper-triangular matrix
//'
//' @param n Number of rows / columns of matrix
//' @param x Values to fill upper-triangle with
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat upperTriFill_cpp(int n, arma::vec x) {
  arma::mat V = eye<mat>(n,n);

  // make empty matrices
  arma::mat Z(n,n,fill::zeros);
  arma::mat X(n,n,fill::zeros);

  // fill matrices with integers
  arma::vec idx = linspace<mat>(1,n,n);
  X.each_col() += idx;
  Z.each_row() += trans(idx);

  // assign upper triangular elements
  // the >= allows inclusion of diagonal elements
  V.elem(find(Z>=X)) = x;

  return(V);
}

//' @title Taylor Series Estimate of Covariance Matrix for Partial Correlations
//'
//' @description This function calculates a second-order Taylor Series estimate of the covariance matrix for partial correlations of a weakly stationary multivariate time series.
//'
//' @param ts \eqn{nt} x \eqn{p} matrix of observed \eqn{p}-variate time series.
//' @param bw nonnegative bandwidth parameter.
//' @param iMatq matrix of indices for partial correlations equal to unique(royVarhelper(p)[, 1:2]).
//' @param iMate matrix of indices for partial correlations equal to royVarhelper(p, errors = TRUE).
//' @param q number of unique partial correlations equal to choose(\eqn{p}, 2).
//'
//' @return \eqn{q} x \eqn{q} covariance matrix
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat partialCov_cpp(arma::mat ts, int bw, arma::mat iMatq, arma::mat iMate, int q) {

  iMatq = iMatq - 1;
  iMate = iMate - 1;

  int p = ts.n_cols;
  int ncovs = iMate.n_rows;
  int N = ts.n_rows;

  // Initialize pcCovs with zeros
  arma::vec pcCovs(ncovs, arma::fill::zeros);

  int bw2 = pow(bw + 1, 2);
  int n2bw = ceil((N - 2*bw)/2);

  // Tapering weights
  arma::vec hu2s = cosTaper_cpp(regspace<vec>(-bw, bw)).elem(regspace<uvec>(bw - 1, 2*bw - 1));
  hu2s(0) = 1;

  // Calculating residuals and tapered sub-matrices
  arma::mat resids(N, 2*q);
  arma::cube tapeSubsii(bw2, bw2, 2*q);
  arma::cube tapeSubsij(bw2, bw2, 2*q);
  int i;
  int j;
  arma::mat projMat(N, N);
  arma::mat designMat(N, p);

  for(int iter = 1; iter < q + 1; iter++) {
    designMat = ts;
    i = iMatq(iter-1, 0);
    j = iMatq(iter-1, 1);
    designMat.shed_col(i);
    designMat.shed_col(j-1);
    projMat = designMat*arma::solve(designMat.t() * designMat, designMat.t());
    resids(span::all, iter*2 - 2) = ts.col(i) - projMat*ts.col(i);
    resids(span::all, iter*2 - 1) = ts.col(j) - projMat*ts.col(j);
    tapeSubsii.slice(iter*2 - 2) = taperCovSub_cpp(resids(span::all, iter*2 - 2), resids(span::all, iter*2 - 2), bw, hu2s);
    tapeSubsii.slice(iter*2 - 1) = taperCovSub_cpp(resids(span::all, iter*2 - 1), resids(span::all, iter*2 - 1), bw, hu2s);
    tapeSubsij.slice(iter*2 - 2) = taperCovSub_cpp(resids(span::all, iter*2 - 2), resids(span::all, iter*2 - 1), bw, hu2s);
    tapeSubsij.slice(iter*2 - 1) = taperCovSub_cpp(resids(span::all, iter*2 - 1), resids(span::all, iter*2 - 2), bw, hu2s);
  }

  double ssi;
  double ssj;
  double ssij;
  double rdenomij;

  double ssk;
  double ssm;
  double sskm;
  double rdenomkm;

  arma::mat h11ij(bw2, bw2, fill::zeros);
  arma::mat h22ij(bw2, bw2, fill::zeros);
  arma::mat h12ij(bw2, bw2, fill::zeros);
  arma::mat h11km(bw2, bw2, fill::zeros);
  arma::mat h22km(bw2, bw2, fill::zeros);
  arma::mat h12km(bw2, bw2, fill::zeros);

  arma::mat sigEpsik(bw2, bw2, fill::zeros);
  arma::mat sigEpsim(bw2, bw2, fill::zeros);
  arma::mat sigEpsjk(bw2, bw2, fill::zeros);
  arma::mat sigEpsjm(bw2, bw2, fill::zeros);

  arma::mat cpse1(bw2, bw2, fill::zeros);
  arma::mat cpse2(bw2, bw2, fill::zeros);
  arma::mat cpse3(bw2, bw2, fill::zeros);
  arma::mat cpse4(bw2, bw2, fill::zeros);

  double Vijkm;

  for(int iter = 0; iter < ncovs; iter++) {
    ssi = tapeSubsii.slice(iMate(iter, 0))(0, 0)*(N-1);
    ssj = tapeSubsii.slice(iMate(iter, 1))(0, 0)*(N-1);
    ssij = tapeSubsij.slice(iMate(iter, 0))(0, 0)*(N-1);
    rdenomij = 1 / sqrt(ssi * ssj);

    ssk = tapeSubsii.slice(iMate(iter, 2))(0, 0)*(N-1);
    ssm = tapeSubsii.slice(iMate(iter, 3))(0, 0)*(N-1);
    sskm = tapeSubsij.slice(iMate(iter, 2))(0, 0)*(N-1);
    rdenomkm = 1 / sqrt(ssk * ssm);

    // Hessian matrices
    h11ij = -2*tapeSubsij.slice(iMate(iter, 0))*pow(rdenomij, 3) * ssj + 3*tapeSubsii.slice(iMate(iter, 0))*pow(rdenomij, 5)*pow(ssj, 2)*ssij - ssij*pow(rdenomij, 3)*ssj*eye(bw2, bw2);
    h22ij = -2*tapeSubsij.slice(iMate(iter, 1))*pow(rdenomij, 3) * ssi + 3*tapeSubsii.slice(iMate(iter, 1))*pow(rdenomij, 5)*pow(ssi, 2)*ssij - ssij*pow(rdenomij, 3)*ssi*eye(bw2, bw2);

    h12ij = -tapeSubsii.slice(iMate(iter, 0))*pow(rdenomij, 3)*ssj + tapeSubsij.slice(iMate(iter, 0))*pow(rdenomij, 3)*ssij - tapeSubsii.slice(iMate(iter, 1))*pow(rdenomij, 3)*ssi + rdenomij*eye(bw2, bw2);

    h11km = -2*tapeSubsij.slice(iMate(iter, 2))*pow(rdenomkm, 3) * ssm + 3*tapeSubsii.slice(iMate(iter, 2))*pow(rdenomkm, 5)*pow(ssm, 2)*sskm - sskm*pow(rdenomkm, 3)*ssm *eye(bw2, bw2);

    h22km = -2*tapeSubsij.slice(iMate(iter, 3))*pow(rdenomkm, 3) * ssk + 3*tapeSubsii.slice(iMate(iter, 3))*pow(rdenomkm, 5)*pow(ssk, 2)*sskm - sskm*pow(rdenomkm, 3)*ssk *eye(bw2, bw2);

    h12km = -tapeSubsii.slice(iMate(iter, 2))*pow(rdenomkm, 3)*ssm + tapeSubsij.slice(iMate(iter, 2))*pow(rdenomkm, 3)*sskm - tapeSubsii.slice(iMate(iter, 3))*pow(rdenomkm, 3)*ssk + rdenomkm*eye(bw2, bw2);

    sigEpsik = taperCovSub_cpp(resids(span::all,iMate(iter, 0)), resids(span::all,iMate(iter, 2)), bw, hu2s);
    sigEpsim = taperCovSub_cpp(resids(span::all,iMate(iter, 0)), resids(span::all,iMate(iter, 3)), bw, hu2s);
    sigEpsjk = taperCovSub_cpp(resids(span::all,iMate(iter, 1)), resids(span::all,iMate(iter, 2)), bw, hu2s);
    sigEpsjm = taperCovSub_cpp(resids(span::all,iMate(iter, 1)), resids(span::all,iMate(iter, 3)), bw, hu2s);

    cpse1 = eigenMult2((eigenMult2(h11ij, sigEpsik) + eigenMult2(h12ij, sigEpsjk)),
                       (eigenMult2(h11km, sigEpsik) + eigenMult2(h12km, sigEpsim)));
    cpse2 = eigenMult2((eigenMult2(h11ij, sigEpsim) + eigenMult2(h12ij, sigEpsjm)),
                       (eigenMult2(h12km, sigEpsik) + eigenMult2(h22km, sigEpsim)));
    cpse3 = eigenMult2((eigenMult2(h12ij, sigEpsik) + eigenMult2(h22ij, sigEpsjk)),
                       (eigenMult2(h11km, sigEpsjk) + eigenMult2(h12km, sigEpsjm)));
    cpse4 = eigenMult2((eigenMult2(h12ij, sigEpsim) + eigenMult2(h22ij, sigEpsjm)),
                       (eigenMult2(h12km, sigEpsjk) + eigenMult2(h22km, sigEpsjm)));

    Vijkm = n2bw*cpse1(bw, bw) + accu(cpse1(span(0, bw-1), span(0, bw-1)).diag()) +
      n2bw*cpse2(bw, bw) + accu(cpse2(span(0, bw-1), span(0, bw-1)).diag()) +
      n2bw*cpse3(bw, bw) + accu(cpse3(span(0, bw-1), span(0, bw-1)).diag()) +
      n2bw*cpse4(bw, bw) + accu(cpse4(span(0, bw-1), span(0, bw-1)).diag());

    pcCovs(iter) = Vijkm;
  }

  // Instantiating covariance matrix
  arma::mat pcCovMat = upperTriFill_cpp(q, pcCovs);
  pcCovMat.diag() = pcCovMat.diag() / 2;

  return(pcCovMat + pcCovMat.t());
}

//' @title Taylor Series Estimate of Covariance Matrix for Partial Correlations
//'
//' @description This function calculates a second-order Taylor Series estimate of the covariance matrix for partial correlations of a weakly stationary multivariate time series.
//'
//' @param mvts \eqn{nt} x \eqn{p} matrix of observed \eqn{p}-variate time series.
//' @param bandwidth nonnegative bandwidth parameter.
//' @param structure Optional covariance structure, indicating whether to use a covariance estimator in which every entry is estimated ("unstructured"), covariances between correlations with disjoint pairs of variables are set to 0 while others are estimated ("intersecting-pairs"), or off-diagonal entries are set to 0 and diagonal entries are estimated ("diagonal").
//' @param correlation_indices matrix of indices for partial correlations equal to unique(royVarhelper(p)[, 1:2]).
//' @param residual_pairs matrix of indices for residual pairs equal to royVarhelper(p, errors = TRUE).
//' @param q number of unique partial correlations equal to choose(\eqn{p}, 2).
//' @param correlation_pairs (choose(\eqn{q}, 2) + \eqn{q}) x 4 matrix of indices for partial correlations equal to royVarhelper(p, errors = FALSE).
//' @param residuals_matrix \eqn{nt} x \eqn{2q} matrix of empirical residuals (optional). If not specified, residuals are obtained using ordinary least squares.
//'
//' @return \eqn{q} x \eqn{q} covariance matrix
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
 arma::mat partial_corr_asymptotic_cov_cpp(arma::mat mvts, int bandwidth, std::string structure, int q, arma::mat correlation_indices, arma::mat residual_pairs, arma::mat correlation_pairs, Nullable<NumericMatrix> residuals = R_NilValue) {

   correlation_indices = correlation_indices - 1;
   residual_pairs = residual_pairs - 1;

   int p = mvts.n_cols;
   int ncovs = residual_pairs.n_rows;
   // int ncovs = Rcpp::as<int>(Rcpp::wrap(R::choose(q, 2))) + q;
   int N = mvts.n_rows;

   // Initialize pcCovs with zeros
   arma::vec pcCovs(ncovs, arma::fill::zeros);

   int bandwidth2 = pow(bandwidth + 1, 2);
   int n2bandwidth = ceil((N - 2*bandwidth)/2);

   // Tapering weights
   arma::vec hu2s = cosTaper_cpp(regspace<vec>(-bandwidth, bandwidth)).elem(regspace<uvec>(bandwidth - 1, 2*bandwidth - 1));
   hu2s(0) = 1;

   // Calculating tapered sub-matrices
   arma::cube tapeSubsii(bandwidth2, bandwidth2, 2*q);
   arma::cube tapeSubsij(bandwidth2, bandwidth2, 2*q);
   int i;
   int j;
   arma::mat projMat(N, N);
   arma::mat designMat(N, p);

   // Instantiate matrix for residuals
   arma::mat residuals_matrix(N, 2*q);

   // Check if residuals is specified
   if (residuals.isNull()){
     // Calculate matrix of residuals if needed
     for(int iter = 1; iter < q + 1; iter++) {
       designMat = mvts;
       i = correlation_indices(iter-1, 0);
       j = correlation_indices(iter-1, 1);
       designMat.shed_col(i);
       designMat.shed_col(j-1);
       projMat = designMat*arma::solve(designMat.t() * designMat, designMat.t());
       residuals_matrix(span::all, iter*2 - 2) = mvts.col(i) - projMat*mvts.col(i);
       residuals_matrix(span::all, iter*2 - 1) = mvts.col(j) - projMat*mvts.col(j);
       tapeSubsii.slice(iter*2 - 2) = taperCovSub_cpp(residuals_matrix(span::all, iter*2 - 2), residuals_matrix(span::all, iter*2 - 2), bandwidth, hu2s);
       tapeSubsii.slice(iter*2 - 1) = taperCovSub_cpp(residuals_matrix(span::all, iter*2 - 1), residuals_matrix(span::all, iter*2 - 1), bandwidth, hu2s);
       tapeSubsij.slice(iter*2 - 2) = taperCovSub_cpp(residuals_matrix(span::all, iter*2 - 2), residuals_matrix(span::all, iter*2 - 1), bandwidth, hu2s);
       tapeSubsij.slice(iter*2 - 1) = taperCovSub_cpp(residuals_matrix(span::all, iter*2 - 1), residuals_matrix(span::all, iter*2 - 2), bandwidth, hu2s);
     }
   } else{
     // Convert residuals to arma::mat
     NumericMatrix residuals_(residuals);
     residuals_matrix = arma::mat(residuals_.begin(), residuals_.nrow(), residuals_.ncol(), false);
     for(int iter = 1; iter < q + 1; iter++) {
       i = correlation_indices(iter-1, 0);
       j = correlation_indices(iter-1, 1);
       tapeSubsii.slice(iter*2 - 2) = taperCovSub_cpp(residuals_matrix(span::all, iter*2 - 2), residuals_matrix(span::all, iter*2 - 2), bandwidth, hu2s);
       tapeSubsii.slice(iter*2 - 1) = taperCovSub_cpp(residuals_matrix(span::all, iter*2 - 1), residuals_matrix(span::all, iter*2 - 1), bandwidth, hu2s);
       tapeSubsij.slice(iter*2 - 2) = taperCovSub_cpp(residuals_matrix(span::all, iter*2 - 2), residuals_matrix(span::all, iter*2 - 1), bandwidth, hu2s);
       tapeSubsij.slice(iter*2 - 1) = taperCovSub_cpp(residuals_matrix(span::all, iter*2 - 1), residuals_matrix(span::all, iter*2 - 2), bandwidth, hu2s);
     }
   }

   double ssi;
   double ssj;
   double ssij;
   double rdenomij;

   double ssk;
   double ssm;
   double sskm;
   double rdenomkm;

   arma::mat h11ij(bandwidth2, bandwidth2, fill::zeros);
   arma::mat h22ij(bandwidth2, bandwidth2, fill::zeros);
   arma::mat h12ij(bandwidth2, bandwidth2, fill::zeros);
   arma::mat h11km(bandwidth2, bandwidth2, fill::zeros);
   arma::mat h22km(bandwidth2, bandwidth2, fill::zeros);
   arma::mat h12km(bandwidth2, bandwidth2, fill::zeros);

   arma::mat sigEpsik(bandwidth2, bandwidth2, fill::zeros);
   arma::mat sigEpsim(bandwidth2, bandwidth2, fill::zeros);
   arma::mat sigEpsjk(bandwidth2, bandwidth2, fill::zeros);
   arma::mat sigEpsjm(bandwidth2, bandwidth2, fill::zeros);

   arma::mat cpse1(bandwidth2, bandwidth2, fill::zeros);
   arma::mat cpse2(bandwidth2, bandwidth2, fill::zeros);
   arma::mat cpse3(bandwidth2, bandwidth2, fill::zeros);
   arma::mat cpse4(bandwidth2, bandwidth2, fill::zeros);

   double Vijkm;

   for(int iter = 0; iter < ncovs; iter++) {

     // Make covariance between correlations be 0 if structure == "diagonal" (i != k or j != m)
     if (structure == "diagonal" && (correlation_pairs(iter, 0) != correlation_pairs(iter, 2) || correlation_pairs(iter, 1) != correlation_pairs(iter, 3))) {
       continue; // Skip to the next iteration and leave as 0
     }

     // Make covariance between disjoint pairs be 0 if structure == "intersecting-pairs" (i != k,m and j != k,m)
     if (structure == "intersecting-pairs" && correlation_pairs(iter, 0) != correlation_pairs(iter, 2) && correlation_pairs(iter, 0) != correlation_pairs(iter, 3) &&
         correlation_pairs(iter, 1) != correlation_pairs(iter, 2) && correlation_pairs(iter, 1) != correlation_pairs(iter, 3)) {
       continue; // Skip to the next iteration and leave as 0
     }

     ssi = tapeSubsii.slice(residual_pairs(iter, 0))(0, 0)*(N-1);
     ssj = tapeSubsii.slice(residual_pairs(iter, 1))(0, 0)*(N-1);
     ssij = tapeSubsij.slice(residual_pairs(iter, 0))(0, 0)*(N-1);
     rdenomij = 1 / sqrt(ssi * ssj);

     ssk = tapeSubsii.slice(residual_pairs(iter, 2))(0, 0)*(N-1);
     ssm = tapeSubsii.slice(residual_pairs(iter, 3))(0, 0)*(N-1);
     sskm = tapeSubsij.slice(residual_pairs(iter, 2))(0, 0)*(N-1);
     rdenomkm = 1 / sqrt(ssk * ssm);

     // Hessian matrices
     h11ij = -2*tapeSubsij.slice(residual_pairs(iter, 0))*pow(rdenomij, 3) * ssj + 3*tapeSubsii.slice(residual_pairs(iter, 0))*pow(rdenomij, 5)*pow(ssj, 2)*ssij - ssij*pow(rdenomij, 3)*ssj*eye(bandwidth2, bandwidth2);
     h22ij = -2*tapeSubsij.slice(residual_pairs(iter, 1))*pow(rdenomij, 3) * ssi + 3*tapeSubsii.slice(residual_pairs(iter, 1))*pow(rdenomij, 5)*pow(ssi, 2)*ssij - ssij*pow(rdenomij, 3)*ssi*eye(bandwidth2, bandwidth2);

     h12ij = -tapeSubsii.slice(residual_pairs(iter, 0))*pow(rdenomij, 3)*ssj + tapeSubsij.slice(residual_pairs(iter, 0))*pow(rdenomij, 3)*ssij - tapeSubsii.slice(residual_pairs(iter, 1))*pow(rdenomij, 3)*ssi + rdenomij*eye(bandwidth2, bandwidth2);

     h11km = -2*tapeSubsij.slice(residual_pairs(iter, 2))*pow(rdenomkm, 3) * ssm + 3*tapeSubsii.slice(residual_pairs(iter, 2))*pow(rdenomkm, 5)*pow(ssm, 2)*sskm - sskm*pow(rdenomkm, 3)*ssm *eye(bandwidth2, bandwidth2);

     h22km = -2*tapeSubsij.slice(residual_pairs(iter, 3))*pow(rdenomkm, 3) * ssk + 3*tapeSubsii.slice(residual_pairs(iter, 3))*pow(rdenomkm, 5)*pow(ssk, 2)*sskm - sskm*pow(rdenomkm, 3)*ssk *eye(bandwidth2, bandwidth2);

     h12km = -tapeSubsii.slice(residual_pairs(iter, 2))*pow(rdenomkm, 3)*ssm + tapeSubsij.slice(residual_pairs(iter, 2))*pow(rdenomkm, 3)*sskm - tapeSubsii.slice(residual_pairs(iter, 3))*pow(rdenomkm, 3)*ssk + rdenomkm*eye(bandwidth2, bandwidth2);

     sigEpsik = taperCovSub_cpp(residuals_matrix(span::all,residual_pairs(iter, 0)), residuals_matrix(span::all,residual_pairs(iter, 2)), bandwidth, hu2s);
     sigEpsim = taperCovSub_cpp(residuals_matrix(span::all,residual_pairs(iter, 0)), residuals_matrix(span::all,residual_pairs(iter, 3)), bandwidth, hu2s);
     sigEpsjk = taperCovSub_cpp(residuals_matrix(span::all,residual_pairs(iter, 1)), residuals_matrix(span::all,residual_pairs(iter, 2)), bandwidth, hu2s);
     sigEpsjm = taperCovSub_cpp(residuals_matrix(span::all,residual_pairs(iter, 1)), residuals_matrix(span::all,residual_pairs(iter, 3)), bandwidth, hu2s);

     cpse1 = eigenMult2((eigenMult2(h11ij, sigEpsik) + eigenMult2(h12ij, sigEpsjk)),
                        (eigenMult2(h11km, sigEpsik) + eigenMult2(h12km, sigEpsim)));
     cpse2 = eigenMult2((eigenMult2(h11ij, sigEpsim) + eigenMult2(h12ij, sigEpsjm)),
                        (eigenMult2(h12km, sigEpsik) + eigenMult2(h22km, sigEpsim)));
     cpse3 = eigenMult2((eigenMult2(h12ij, sigEpsik) + eigenMult2(h22ij, sigEpsjk)),
                        (eigenMult2(h11km, sigEpsjk) + eigenMult2(h12km, sigEpsjm)));
     cpse4 = eigenMult2((eigenMult2(h12ij, sigEpsim) + eigenMult2(h22ij, sigEpsjm)),
                        (eigenMult2(h12km, sigEpsjk) + eigenMult2(h22km, sigEpsjm)));

     Vijkm = n2bandwidth*cpse1(bandwidth, bandwidth) + accu(cpse1(span(0, bandwidth-1), span(0, bandwidth-1)).diag()) +
             n2bandwidth*cpse2(bandwidth, bandwidth) + accu(cpse2(span(0, bandwidth-1), span(0, bandwidth-1)).diag()) +
             n2bandwidth*cpse3(bandwidth, bandwidth) + accu(cpse3(span(0, bandwidth-1), span(0, bandwidth-1)).diag()) +
             n2bandwidth*cpse4(bandwidth, bandwidth) + accu(cpse4(span(0, bandwidth-1), span(0, bandwidth-1)).diag());

     pcCovs(iter) = Vijkm;
   }

   // Instantiating covariance matrix
   arma::mat pcCovMat = upperTriFill_cpp(q, pcCovs);
   pcCovMat.diag() = pcCovMat.diag() / 2;

   return(pcCovMat + pcCovMat.t());
 }

//' @title Asymptotic Covariance Matrix for Correlations of Hedges and Olkin (1983)
//'
//' @description This function calculates an asymptotic estimate of the covariance matrix for a vector of correlations as given by Hedges and Olkin (1983).
//'
//' @param correlation_matrix \eqn{p} x \eqn{p} correlation matrix
//' @param n number of observations
//' @param structure Optional covariance structure, indicating whether to use a covariance estimator in which every entry is estimated ("unstructured"), covariances between correlations with disjoint pairs of variables are set to 0 while others are estimated ("intersecting-pairs"), or off-diagonal entries are set to 0 and diagonal entries are estimated ("diagonal").
//' @param correlation_pairs (choose(\eqn{q}, 2) + \eqn{q}) x 4 matrix of indices for correlations equal to royVarhelper(p, errors = FALSE) where \eqn{q} = choose(\eqn{p}, 2) .
//'
//' @return \eqn{q} x \eqn{q} covariance matrix
//'
//' @references
//' Hedges, L. V. and Olkin, I. (1983). Joint distributions of some indices based on correlation coefficients. In Studies in Econometrics, Time Series, and Multivariate Statistics, 437–454. Academic Press.
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
 arma::mat correlations_covariance_hedges_olkin_cpp(arma::mat correlation_matrix, int n, arma::mat correlation_pairs, std::string structure = "unstructured") {

   // Adjusting indices to start indexing at 0 for C++
   correlation_pairs = correlation_pairs - 1;

   // Number of variables
   int p = correlation_matrix.n_cols;

   // Number of unique correlations
   int q = (p * (p - 1)) / 2;

   // Number of unique covariances to calculate
   int n_covs = correlation_pairs.n_rows;

   // Initialize correlation covariances with zeros
   arma::vec correlation_covariances(n_covs, arma::fill::zeros);

   double covariance_ijkm;
   int i;
   int j;
   int k;
   int m;

   double rij;
   double rik;
   double rim;
   double rjk;
   double rjm;
   double rkm;

   for(int iter = 0; iter < n_covs; iter++) {

     // Make covariance between correlations be 0 if structure == "diagonal" (i != k or j != m)
     if (structure == "diagonal" && (correlation_pairs(iter, 0) != correlation_pairs(iter, 2) || correlation_pairs(iter, 1) != correlation_pairs(iter, 3))) {
       continue; // Skip to the next iteration and leave as 0
     }

     // Make covariance between disjoint pairs be 0 if structure == "intersecting-pairs" (i != k,m and j != k,m)
     if (structure == "intersecting-pairs" && correlation_pairs(iter, 0) != correlation_pairs(iter, 2) && correlation_pairs(iter, 0) != correlation_pairs(iter, 3) &&
         correlation_pairs(iter, 1) != correlation_pairs(iter, 2) && correlation_pairs(iter, 1) != correlation_pairs(iter, 3)) {
       continue; // Skip to the next iteration and leave as 0
     }

     i = correlation_pairs(iter, 0);
     j = correlation_pairs(iter, 1);
     k = correlation_pairs(iter, 2);
     m = correlation_pairs(iter, 3);

     rij = correlation_matrix(i, j);
     rik = correlation_matrix(i, k);
     rim = correlation_matrix(i, m);
     rjk = correlation_matrix(j, k);
     rjm = correlation_matrix(j, m);
     rkm = correlation_matrix(k, m);

     covariance_ijkm = 0.5 * rij * rkm * (rik*rik + rim*rim + rjk*rjk + rjm*rjm) +
                             rik * rjm + rim * rjk -
                            (rij * rik * rim + rij * rjk * rjm + rik * rjk * rkm + rim * rjm * rkm);

     correlation_covariances(iter) = covariance_ijkm;
   }

   // Instantiating covariance matrix
   arma::mat covariance_matrix = upperTriFill_cpp(q, correlation_covariances / n);
   covariance_matrix.diag() = covariance_matrix.diag() / 2;

   return(covariance_matrix + covariance_matrix.t());
 }

// [[Rcpp::export]]
double thetaHat_cpp(int i, int j, int l, int m, arma::mat ts, int n, arma::vec hu2s, arma::cube ccMat) {
  arma::vec vals1(n, fill::zeros);
  arma::vec vals2(n, fill::zeros);
  for(int h = 0; h < n; h++) {
    vals1(h) = ccMat(h, i, j);
    vals2(h) = ccMat(h, l, m);
  }
  return(sum(hu2s % vals1 % vals2));
}

// [[Rcpp::export]]
double deltaHat_cpp(int i, int j, int l, int m, arma::mat mvts, int n, arma::vec hu2s,
                    arma::vec ccs, arma::cube ccMat) {
  double crossProd = ccs(i)*ccs(j)*ccs(l)*ccs(m);
  return(thetaHat_cpp(i, j, l, m, mvts, n, hu2s, ccMat) / sqrt(crossProd));
}

//' @title Roy Asymptotic Variance
//'
//' @description This function calculates the asymptotic covariance matrix for correlations of a stationary multivariate time series as derived by Roy (1989).
//'
//' @param iMat Matrix of correlation indices
//' @param tsData Matrix of observed \eqn{n}-length \eqn{p}-variate time series
//' @param q Integer equal to the number of unique variables pairs given by choose(\eqn{p}, 2)
//' @param bw Bandwidth parameter
//'
//' @return \eqn{q} x \eqn{q} covariance matrix
//'
//' @references Roy, R. (1989). Asymptotic covariance structure of serial correlations in multivariate time series. Biometrika, 76(4), 824-827.
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat royVar_cpp(arma::mat iMat, arma::mat tsData, int q, int bw = 10) {
  int N = iMat.n_rows;
  int n = tsData.n_rows;
  int p = tsData.n_cols;

  arma::mat pcMat = corrMat_cpp(tsData, false);

  // Creating hu2s for thetaHat_cpp
  arma::vec mySeq = regspace<vec>((-n+1), (n-1)) - 1;
  arma::vec non0 = mySeq(abs(mySeq) <= bw);
  arma::vec hu2s = pow(cosTaper_cpp(non0), 2.0);
  int uLength = non0.n_elem;

  // Calculating crossCov_cpp for lag 0
  arma::vec ccs(p);
  for(int i = 0; i < p; i++) {
    ccs(i) = crossCov_cpp(0, tsData.col(i), tsData.col(i));
  }

  // Calculating crossCov_cpp for different lags
  arma::cube ccMat(uLength, p, p);
  for(int h = 0; h < uLength; h++) {
    for(int i = 0; i < p; i++) {
      for(int j = 0; j < p; j++) {
        ccMat(h, i, j) = crossCov_cpp(non0[h], tsData.col(i), tsData.col(j));
      }
    }
  }

  arma::vec royCov(N);
  for(int k = 0; k < N; k++) {
    int a = iMat(k, 0) - 1;
    int b = iMat(k, 1) - 1;
    int d = iMat(k, 2) - 1;
    int e = iMat(k, 3) - 1;
    royCov[k] = (0.50*pcMat(a, b)*pcMat(d, e)*(deltaHat_cpp(a, d, a, d, tsData,
                                        uLength, hu2s, ccs, ccMat) +
                                          deltaHat_cpp(a, e, a, e, tsData, uLength, hu2s, ccs, ccMat) +
                                          deltaHat_cpp(b, d, b, d, tsData, uLength, hu2s, ccs, ccMat) +
                                          deltaHat_cpp(b, e, b, e, tsData, uLength, hu2s, ccs, ccMat))
                   - pcMat(a, b)*(deltaHat_cpp(a, d, a, e, tsData, uLength, hu2s, ccs, ccMat) +
                     deltaHat_cpp(b, d, b, e, tsData, uLength, hu2s, ccs, ccMat))
                   - pcMat(d, e)*(deltaHat_cpp(b, d, a, d, tsData, uLength, hu2s, ccs, ccMat) +
                     deltaHat_cpp(b, e, a, e, tsData, uLength, hu2s, ccs, ccMat)) +
                     deltaHat_cpp(a, d, b, e, tsData, uLength, hu2s, ccs, ccMat) +
                     deltaHat_cpp(b, d, a, e, tsData, uLength, hu2s, ccs, ccMat));
  }

  arma::mat upperMat = upperTriFill_cpp(q, royCov / n);
  arma::vec diagVals = upperMat.diag();
  arma::mat lowerMat = upperMat.t();
  arma::mat ret = upperMat + lowerMat;
  ret.diag()  = diagVals;
  return(ret);
}

//' @export
// [[Rcpp::export]]
arma::mat royVar2_cpp(arma::mat iMat, arma::mat tsData, int q, int bw = 10) {
  int N = iMat.n_rows;
  int n = tsData.n_rows;
  int p = tsData.n_cols;

  std::cout << "iMat Rows: " << N << ", tsData Rows: " << n << ", tsData Cols: " << p << std::endl;

  arma::mat pcMat = corrMat_cpp(tsData, false);
  std::cout << "pcMat calculated." << std::endl;

  // Creating hu2s for thetaHat_cpp
  arma::vec mySeq = regspace<vec>((-n+1), (n-1)) - 1;
  arma::vec non0 = mySeq(abs(mySeq) <= bw);
  arma::vec hu2s = pow(cosTaper_cpp(non0), 2.0);
  int uLength = non0.n_elem;

  std::cout << "mySeq: " << mySeq.t() << std::endl;
  std::cout << "non0: " << non0.t() << std::endl;
  std::cout << "uLength: " << uLength << std::endl;

  // Calculating crossCov_cpp for lag 0
  arma::vec ccs(p);
  for(int i = 0; i < p; i++) {
    ccs(i) = crossCov_cpp(0, tsData.col(i), tsData.col(i));
  }

  std::cout << "ccs: " << ccs.t() << std::endl;

  // Calculating crossCov_cpp for different lags
  arma::cube ccMat(uLength, p, p);
  for(int h = 0; h < uLength; h++) {
    for(int i = 0; i < p; i++) {
      for(int j = 0; j < p; j++) {
        ccMat(h, i, j) = crossCov_cpp(non0[h], tsData.col(i), tsData.col(j));
      }
    }
  }

  std::cout << "ccMat calculated." << std::endl;

  arma::vec royCov(N);
  for(int k = 0; k < N; k++) {
    int a = iMat(k, 0) - 1;
    int b = iMat(k, 1) - 1;
    int d = iMat(k, 2) - 1;
    int e = iMat(k, 3) - 1;

    std::cout << "Indices a: " << a << ", b: " << b << ", d: " << d << ", e: " << e << std::endl;

    royCov[k] = (0.50 * pcMat(a, b) * pcMat(d, e) * (deltaHat_cpp(a, d, a, d, tsData,
                                            uLength, hu2s, ccs, ccMat) +
                                              deltaHat_cpp(a, e, a, e, tsData, uLength, hu2s, ccs, ccMat) +
                                              deltaHat_cpp(b, d, b, d, tsData, uLength, hu2s, ccs, ccMat) +
                                              deltaHat_cpp(b, e, b, e, tsData, uLength, hu2s, ccs, ccMat))
                   - pcMat(a, b) * (deltaHat_cpp(a, d, a, e, tsData, uLength, hu2s, ccs, ccMat) +
                     deltaHat_cpp(b, d, b, e, tsData, uLength, hu2s, ccs, ccMat))
                   - pcMat(d, e) * (deltaHat_cpp(b, d, a, d, tsData, uLength, hu2s, ccs, ccMat) +
                     deltaHat_cpp(b, e, a, e, tsData, uLength, hu2s, ccs, ccMat)) +
                     deltaHat_cpp(a, d, b, e, tsData, uLength, hu2s, ccs, ccMat) +
                     deltaHat_cpp(b, d, a, e, tsData, uLength, hu2s, ccs, ccMat));

    if (k % 10 == 0) {  // Print every 10 iterations
      std::cout << "royCov[" << k << "]: " << royCov[k] << std::endl;
    }
  }

  arma::mat upperMat = upperTriFill_cpp(q, royCov / n);
  std::cout << "upperMat calculated." << std::endl;

  arma::vec diagVals = upperMat.diag();
  arma::mat lowerMat = upperMat.t();
  arma::mat ret = upperMat + lowerMat;
  ret.diag()  = diagVals;

  std::cout << "Final result calculated." << std::endl;
  return(ret);
}

//' Construct Block-Diagonal Matrix
//'
//' @param array3d An n x p x k 3D array of matrices to make into a single block-diagonal matrix
//'
//' @return kn x kp block-diagonal matrix
//'
//' @author
//' Andrew DiLernia
//'
//' @export
// [[Rcpp::export]]
arma::mat bdiagArray_cpp(arma::cube array3d) {

  // Extracting dimensions
  int dim1 = array3d.slice(0).n_rows;
  int dim2 = array3d.slice(0).n_cols;
  int dim3 = array3d.n_slices;

  arma::mat X(dim1*dim3, dim2*dim3);
  X.fill(0.0);

  for(int i=0; i<dim3; i++) {
    X(span(i*dim1, (i+1)*dim1 - 1), span(i*dim2, (i+1)*dim2 - 1)) = array3d.slice(i);
  }

  return(X);
}

// [[Rcpp::export]]
arma::mat xMaker_cpp(int K, int q) {
  arma::mat X(K*q, q);
  arma::mat Xsub = eye<mat>(q, q);
  for(int i = 0; i < K; i++) {
    X.submat(span(i*q, i*q + q-1), span(0, q-1)) = Xsub;
  }
  return(X);
}

//' @export
// [[Rcpp::export]]
arma::mat arrayMean_cpp(arma::cube array3d) {

  // Extracting dimensions
  int dim1 = array3d.slice(0).n_rows;
  int dim2 = array3d.slice(0).n_cols;
  int dim3 = array3d.n_slices;

  arma::mat avgMat(dim1, dim2);
  avgMat.fill(0.0);

  for (int k = 0; k < dim3; k++) {
    // Add matrix `k` from `array3d` e.g. `array3d[ , , k]`
    avgMat = avgMat + array3d.slice(k);
  }

  return(avgMat / dim3);
}

//' @export
// [[Rcpp::export]]
arma::vec upperTri_cpp(arma::mat m, bool incDiag = false) {
  int n = m.n_cols;
  arma::mat V = eye<mat>(n,n);

  // make empty matrices
  arma::mat Z(n,n,fill::zeros);
  arma::mat X(n,n,fill::zeros);

  // fill matrices with integers
  arma::vec idx = linspace<mat>(1,n,n);
  X.each_col() += idx;
  Z.each_row() += trans(idx);
  arma::ivec inds(n*n);

  // Extract upper triangular elements
  if(incDiag){
    return(m.elem(find(Z >= X)));
  } else{
    return(m.elem(find(Z > X)));
  }
}

// [[Rcpp::export]]
arma::cube thetaUpdate_cpp(arma::mat eVec, int K, int q) {

  arma::cube myArray(q, q, K);
  myArray.fill(0.0);

  for (int i = 0; i < K; i++) {
    myArray.slice(i) = eVec.rows(i*q, (i+1)*q - 1) * (eVec.rows(i*q, (i+1)*q - 1).t());
  }

  return(myArray);
}

// [[Rcpp::export]]
arma::mat sigPsiInv_cpp(arma::cube sigmas, double sigVal, int qK, int q, int K) {

  arma::mat sigPsiInv(qK, qK);
  arma::mat psi = eye<mat>(q, q)*sigVal;
  sigPsiInv.fill(0.0);

  for (int i = 0; i < K; i++) {
    sigPsiInv(span(i*q, (i+1)*q - 1), span(i*q, (i+1)*q - 1)) = arma::inv(sigmas.slice(i) + psi);
  }

  return(sigPsiInv);
}

//' @export
 // [[Rcpp::export]]
 arma::field<arma::mat> eigen_decomposition_array_cpp(arma::cube array3d) {

   // Extracting dimensions
   int dim3 = array3d.n_slices;

   arma::field<arma::mat> eigList(dim3 * 2);
   arma::vec eigval;
   arma::mat eigvec;

   for (int k = 0; k < dim3; k++) {
     if (!eig_sym(eigval, eigvec, array3d.slice(k))) {
       Rcpp::stop("Eigen decomposition failed for slice %d", k);
     }
     eigList(k * 2) = arma::mat(eigval); // Explicitly convert eigenvalues to matrix
     eigList(k * 2 + 1) = eigvec;
   }

   return eigList;
 }

//' @export
// [[Rcpp::export]]
arma::field<arma::mat> eigen_cpp(arma::mat myMat) {

  arma::field<arma::mat> eigList(2);
  arma::vec eigval;
  arma::mat eigvec;

  assert(eig_sym(eigval, eigvec, myMat));
  eigList(0) = eigval;
  eigList(1) = eigvec;

  return(eigList);
}

// [[Rcpp::export]]
arma::cube sigPsiInvBlks_cpp(arma::field<arma::mat> eigs, double sigVal, int q, int K) {

  arma::cube sigPsiInv(q, q, K);
  arma::mat lambdaMat(q, q);
  lambdaMat.fill(0.0);
  sigPsiInv.fill(0.0);

  for (int i = 0; i < K; i++) {
    lambdaMat = diagmat(1 / (eigs(i*2) + sigVal));
    sigPsiInv.slice(i) = eigs(i*2+1) * lambdaMat * eigs(i*2+1).t();
  }

  return(sigPsiInv);
}

// [[Rcpp::export]]
arma::mat XtSX_cpp(arma::cube blocks, int q, int K) {

  arma::mat XtSX(q, q);
  XtSX.fill(0.0);

  for (int i = 0; i < K; i++) {
    XtSX = XtSX + blocks.slice(i);
  }

  return(XtSX);
}

//' @title Variance Components Model
//'
//' @description This function implements the variance components model proposed by Fiecas et al. (2017).
//'
//' @param rs Column vector containing q x K unique marginal or partial correlations.
//' @param sigmas 3D array of K estimated q x q covariance matrices for correlations.
//' @param sigEigs List of K matrices containing eigen decomposition matrices for covariance matrices contained in sigmas.
//' @param delta Threshold for algorithm
//' @param maxIters Maximum number of iterations for algorithm
//' @param sig0 Initial value for sigma parameter
//' @param smallRet Logical value, whether or not to return smaller set of results
//'
//' @return List of length 4 containing beta (q x 1), betaCov (q x q), sigma (qK x qK), and psi (qK x qK) estimates. If smallRet = true, then only returns beta and betaCov.
//'
//' @author
//' Andrew DiLernia
//'
//' @references Fiecas, M., Cribben, I., Bahktiari, R., and Cummine, J. (2017). A variance components model for statistical inference on functional connectivity networks. NeuroImage (Orlando, Fla.), 149, 256-266.
//'
//' @export
// [[Rcpp::export]]
List vcm_cpp(arma::mat rs, arma::cube sigmas, arma::field<arma::mat> sigEigs,
                 double delta = 0.001, int maxIters = 100, double sig0 = 0.10, bool smallRet = false) {

  int K = sigmas.n_slices;
  int q = sigmas.slice(0).n_cols;
  int qK = q*K;

  arma::mat sigma = bdiagArray_cpp(sigmas);
  arma::mat sigMean = arrayMean_cpp(sigmas);

  // Creating X diagonal design matrices for each subject
  arma::mat Xs = xMaker_cpp(K, q);

  // Initializing Psi matrix to be diagonal with sig0
  arma::mat iqK = eye<mat>(qK, qK);
  arma::mat psi = iqK*sig0;
  arma::cube sigPsiInvBlks(q, q, K);
  arma::mat XtSX(q, q);
  arma::vec beta(q);
  arma::vec resVec(qK);
  arma::mat thetaMat(q, q);
  arma::mat diffMat(q, q);
  double sigVal = sig0;

  // Variables for convergence
  double epsilon = 1;
  int counter = 0;
  arma::vec betaOld(q);

  // Iterate until convergence
  while(epsilon > delta && counter < maxIters) {

    counter = counter + 1;

    // Updating beta
    sigPsiInvBlks = sigPsiInvBlks_cpp(sigEigs, sigVal, q, K);
    XtSX = XtSX_cpp(sigPsiInvBlks, q, K);
    beta = solve(XtSX, Xs.t() * bdiagArray_cpp(sigPsiInvBlks) * rs);

    // Updating residual vector
    resVec = rs - Xs * beta;

    // Updating theta matrix
    thetaMat = arrayMean_cpp(thetaUpdate_cpp(resVec, K, q));

    // Updating diagonal Psi matrix
    diffMat = thetaMat - sigMean;
    sigVal = mean(diffMat.diag());
    psi = iqK*sigVal;

    // Convergence variables
    epsilon = max(abs(beta - betaOld));
    betaOld = beta;
  }

  // Calculating beta covariance estimate
  sigPsiInvBlks = sigPsiInvBlks_cpp(sigEigs, sigVal, q, K);
  XtSX = XtSX_cpp(sigPsiInvBlks, q, K);
  arma::mat betaCov = arma::inv(XtSX);

  // Returning full or partial results to help with RAM issues
  if(smallRet) {
    List resList = List::create(
      _["beta"] = beta,
      _["betaCov"] = betaCov
    );
    return(resList);
  } else {
    List resList = List::create(
      _["beta"] = beta,
      _["betaCov"] = betaCov,
      _["sigma"] = sigma,
      _["psi"] = psi
    );
    return(resList);
  }
}

// [[Rcpp::export]]
arma::cube listRoyVar_cpp(arma::field<arma::mat> ys, int q, arma::mat iMat) {

  // Extracting dimensions
  int K = ys.n_elem;

  arma::cube royVarArray(q, q, K);

  for (int k = 0; k < K; k++) {
    royVarArray.slice(k) = royVar_cpp(iMat, ys(k), q);
  }

  return(royVarArray);
}

// [[Rcpp::export]]
arma::cube arrayRoyVar_cpp(arma::cube ys, int q, arma::mat iMat) {

  // Extracting dimensions
  int K = ys.n_slices;

  arma::cube royVarArray(q, q, K);

  for (int k = 0; k < K; k++) {
    royVarArray.slice(k) = royVar_cpp(iMat, ys.slice(k), q);
  }

  return(royVarArray);
}

// [[Rcpp::export]]
List royTest_cpp2(arma::field<arma::mat> y1, arma::field<arma::mat> y2, arma::mat iMat,
                  double alpha = 0.05, std::string multAdj = "holm-bonferroni",
                  int nperm = 100) {


  // Creating single list of time series data
  int K1 = y1.size();
  int K2 = y2.size();
  int K = K1 + K2;
  int p = y1(0).n_cols;
  int q = R::choose(p, 2);
  arma::ivec ginds(K);
  ginds.fill(0);

  arma::field<arma::mat> ys(K);
  for(int i = 0; i < K1; ++i)
  {
    ys(i) = y1(i);
  }
  for(int i = 0; i < K2; ++i)
  {
    ys(i + K1) = y2(i);
    ginds(i + K1) = 1;
  }

  arma::cube sigmas = listRoyVar_cpp(ys, q, iMat);
  arma::field<arma::mat> eigDecomps = eigen_decomposition_array_cpp(sigmas);

  List resList = List::create(
    _["ginds"] = ginds,
    _["p"] = p
  );

  return(resList);
}

// [[Rcpp::depends(RcppArmadillo, RcppDist)]]

//' @export
// [[Rcpp::export]]
double dwish_cpp(arma::mat X, int df, arma::mat S,
             bool log_p = false) {

  double ret = dwish(X = X, df = df, S = S, log_p = log_p);

  return(ret);
}

//' @export
// [[Rcpp::export]]
double dwishArray_cpp(arma::cube Xarray, int df, arma::mat S,
                 bool log_p = false) {

  // Extracting dimensions
  int K = Xarray.n_slices;
  double ret = 0;

  for (int k = 0; k < K; k++) {
    arma::mat X = Xarray.slice(k);
    ret = ret + dwish(X = X, df = df, S = S, log_p = log_p);
  }

  return(ret);
}

//' @export
// [[Rcpp::export]]
arma::vec dmvnorm_cpp(arma::mat x, arma::vec mu,
                   arma::mat S, bool log_p = false) {

  arma::vec ret = dmvnorm(x = x, mu = mu, S = S, log_p = log_p);

  return(ret);
}

// [[Rcpp::export]]
arma::uvec customMod(IntegerVector v, int n, int nelems) {
  for(int i = 0; i < nelems; i++) {
    if(v(i) >= n) {
      v(i) = v(i) % n;
    }
  }
  return(as<arma::uvec>(v));
}

// [[Rcpp::export]]
arma::mat subsetRows(arma::mat x, arma::uvec idx) {
  arma::mat xsub;
  xsub = x.rows(idx);
  return(xsub);
}

//' @title Moving Block Bootstrap
//'
//' @description This function implements the moving block bootstrap as proposed by Kunsch (1989).
//'
//' @param mvts \eqn{n} x \eqn{p} matrix of observed \eqn{p}-variate time series
//' @param winLength nonnegative window length parameter
//' @param nBoots Number of bootstrap samples
//'
//' @return 3D array of dimension \eqn{n} x \eqn{p} x nBoots containing the nBoots bootstrap samples
//'
//' @author
//' Andrew DiLernia
//'
//' @references Kunsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations". Annals of Statistics. 17 (3): 1217-1241. doi:10.1214/aos/1176347265.
//'
//' Politis, D. N.; Romano, J. P. (1994). "The Stationary Bootstrap". Journal of the American Statistical Association. 89 (428): 1303-1313. doi:10.1080/01621459.1994.10476870. hdl:10983/25607.
//'
//' @export
// [[Rcpp::export]]
arma::cube blockBoot_cpp(arma::mat mvts, int winLength, int nBoots = 500) {

  // Extracting number of variables and observations
  int p = mvts.n_cols;
  int N = mvts.n_rows;
  int nBlocks;

  // Bootstrap indices
  IntegerVector inds = Rcpp::seq(0, N - winLength);

  // Instantiating vector for block lengths
  IntegerVector winLengths;

    // Number of blocks
    nBlocks = round(N/(winLength));

    // Window lengths (block-size) for each block
    winLengths = rep(winLength, nBlocks);

  // Adjusting so sum of window lengths = N
  winLengths(nBlocks - 1) = winLengths(nBlocks - 1) - (sum(winLengths) - N);

  // Cumulative sum of window lengths
  IntegerVector wlCumSum = cumsum(winLengths);
  wlCumSum.push_front(0);

  //Instantiating vectors for block indices
  IntegerVector bstarts;
  IntegerVector bends;
  arma::uvec binds;

  // Starts of boot indices
  Rcpp::IntegerVector istarts = Rcpp::sample(inds, nBoots*nBlocks, true);

  // Instantiating array for bootstrap samples
  arma::cube bootSamps(N, p, nBoots);

  for(int boot = 0; boot < nBoots; boot++) {
    // Start and end indices for each block
    bstarts = Rcpp::sample(inds, nBlocks, true);
    bends = bstarts + winLengths - 1;

    for(int block = 0; block < nBlocks; block++) {
      // Block indices
      binds = customMod(Rcpp::seq(bstarts(block), bends(block)), N, winLengths(block));
      bootSamps.slice(boot)(span(wlCumSum(block), wlCumSum(block+1) - 1), span::all) = subsetRows(mvts, binds);
    }
  }

  return(bootSamps);
}

//' @title Moving Block Bootstrap for Correlation Coefficients
//'
//' @description This function implements the moving block bootstrap as proposed by Kunsch (1989) for correlation coefficients.
//'
//' @param mvts \eqn{n} x \eqn{p} matrix of observed \eqn{p}-variate time series
//' @param winLength Nonnegative window length parameter
//' @param nBoots Number of bootstrap samples
//' @param stationary Logical value indicating whether to use a variable window length block bootstrap (TRUE) as described by Politis & Romano (1994), or a fixed window length block bootstrap (FALSE) as described by Kunsch (1989)
//' @param partial Logical value indicating whether to implement block bootstrap for the partial (TRUE) or marginal (FALSE) correlation coefficients
//'
//' @return matrix of dimension \eqn{q} x nBoots containing the \eqn{q=} choose(\eqn{p}, 2) correlations for each of the nBoots bootstrap samples
//'
//' @author
//' Andrew DiLernia
//'
//' @references Kunsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations". Annals of Statistics. 17 (3): 1217-1241. doi:10.1214/aos/1176347265.
//'
//' Politis, D. N.; Romano, J. P. (1994). "The Stationary Bootstrap". Journal of the American Statistical Association. 89 (428): 1303-1313. doi:10.1080/01621459.1994.10476870. hdl:10983/25607.
//'
//' @export
// [[Rcpp::export]]
arma::mat blockBootCorr_cpp(arma::mat mvts, int winLength, int nBoots = 500, bool stationary = false, bool partial = true) {

  // Extracting number of variables and observations
  int p = mvts.n_cols;
  int N = mvts.n_rows;
  int nBlocks;

  // Bootstrap indices
  IntegerVector inds = Rcpp::seq(0, N - winLength);

  // Instantiating vector for block lengths
  IntegerVector winLengths;
  double Ndub = N - 1.0;

  if(stationary == true) {
    // Generating variable block lengths
    while(sum(winLengths) < N) {
      winLengths.push_back(std::min(Ndub, R::rgeom(1.0/winLength) + 1.0));
    }

    // Number of blocks
    nBlocks = winLengths.size();
  } else {
    // Number of blocks
    nBlocks = round(N/(winLength));

    // Window lengths (block-size) for each block
    winLengths = rep(winLength, nBlocks);
  }

  // Adjusting so sum of window lengths = N
  winLengths(nBlocks - 1) = winLengths(nBlocks - 1) - (sum(winLengths) - N);

  // Cumulative sum of window lengths
  IntegerVector wlCumSum = cumsum(winLengths);
  wlCumSum.push_front(0);

  //Instantiating vectors for block indices
  IntegerVector bstarts;
  IntegerVector bends;
  arma::uvec binds;

  // Starts of boot indices
  Rcpp::IntegerVector istarts = Rcpp::sample(inds, nBoots*nBlocks, true);

  // Instantiating array for bootstrap samples
  arma::cube bootSamps(N, p, nBoots);

  // Instantiating matrix for bootstrap correlations
  int q = R::choose(p, 2);
  arma::mat bootCorrs(q, nBoots);

  for(int boot = 0; boot < nBoots; boot++) {
    // Start and end indices for each block
    bstarts = Rcpp::sample(inds, nBlocks, true);
    bends = bstarts + winLengths - 1;

    for(int block = 0; block < nBlocks; block++) {
      // Block indices
      binds = customMod(Rcpp::seq(bstarts(block), bends(block)), N, winLengths(block));
      bootSamps.slice(boot)(span(wlCumSum(block), wlCumSum(block+1) - 1), span::all) = subsetRows(mvts, binds);
    }

    bootCorrs(span::all, boot) = upperTri_cpp(corrMat_cpp(bootSamps.slice(boot), partial));
  }

  return(bootCorrs);
}

//' @title Multi-Subject Moving Block Bootstrap for Correlation Coefficients
//'
//' @description This function implements the moving block bootstrap as proposed by Kunsch (1989) for the correlation coefficients of a multi-subject data set.
//'
//' @param mvts 3D array of dimension \eqn{n} x \eqn{p} x \eqn{K} of \eqn{n}-length observed \eqn{p}-variate time series for \eqn{K} individuals
//' @param winLengths nonnegative integer vector of window / block lengths
//' @param nBoots Number of bootstrap samples
//' @param stationary Logical value indicating whether to use a variable window length block bootstrap (TRUE) as described by Politis & Romano (1994), or a fixed window length block bootstrap (FALSE) as described by Kunsch (1989)
//' @param partial Logical value indicating whether to implement block bootstrap for the partial (TRUE) or marginal (FALSE) correlation coefficients
//'
//' @return 3D array of dimension \eqn{q} x nBoots x \eqn{K} containing the \eqn{q=} choose(\eqn{p}, 2) correlations for each of the nBoots bootstrap samples for each of the \eqn{K} individuals
//'
//' @author
//' Andrew DiLernia
//'
//' @references Kunsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations". Annals of Statistics. 17 (3): 1217-1241. doi:10.1214/aos/1176347265.
//'
//' Politis, D. N.; Romano, J. P. (1994). "The Stationary Bootstrap". Journal of the American Statistical Association. 89 (428): 1303-1313. doi:10.1080/01621459.1994.10476870. hdl:10983/25607.
//'
//' @export
// [[Rcpp::export]]
arma::cube multiBlockBootCorr_cpp(arma::cube mvts, IntegerVector winLengths, int nBoots = 500, bool stationary = false, bool partial = true) {

  // Extracting number of variables and participants
  int p = mvts.slice(0).n_cols;
  int K = mvts.n_slices;
  int q = R::choose(p, 2);

  // Instantiating array for bootstrap samples
  arma::cube bootCorrs(q, nBoots, K);

  for(int k = 0; k < K; k++) {
    bootCorrs.slice(k) = blockBootCorr_cpp(mvts.slice(k), winLengths(k), nBoots, stationary, partial);
  }

  return(bootCorrs);
}

