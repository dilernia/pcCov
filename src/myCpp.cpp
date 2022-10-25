#include <cstdlib>
#include <iostream>
#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <wishart.h>
#include <mvnorm.h>
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
  Eigen::MatrixXd C = arma2eigen(A) * arma2eigen(B);

  return eigen2arma(C);
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
  Eigen::MatrixXd Aeig = arma2eigen(A);
  Eigen::MatrixXd Beig = arma2eigen(B);
  Eigen::MatrixXd Ceig = arma2eigen(C);
  Eigen::MatrixXd D = Aeig * Beig * Ceig;

  arma::mat ret = eigen2arma(D);

  return(ret);
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
  Eigen::MatrixXd Aeig = arma2eigen(A);
  Eigen::MatrixXd Beig = arma2eigen(B);
  Eigen::MatrixXd Ceig = arma2eigen(C);
  Eigen::MatrixXd Deig = arma2eigen(D);
  Eigen::MatrixXd E = Aeig * Beig * Ceig * Deig;

  arma::mat ret = eigen2arma(E);
  return(ret);
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
//' @param tsData A data matrix.
//' @param partial Logical. Whether to calculate partial (TRUE) or marginal (FALSE) correlation matrix
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
NumericVector cosTaper_cpp(IntegerVector u){
  IntegerVector uNew = u - min(u);
  NumericVector sqrtRet = sin(M_PI*as<NumericVector>(uNew) / u.size());
  return(sqrtRet * sqrtRet);
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
  IntegerVector non0 = seq(0, banw);
  arma::vec hu2s = cosTaper_cpp(seq(-banw, banw))[(seq(banw, 2*banw) - 1)];
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
  int N = ts1.n_elem;
  IntegerVector non0 = seq(0, banw);
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
//' @param ts \eqn{nt} x \eqn{p} matrix of observed p-variate time series.
//' @param bw nonnegative bandwidth parameter.
//' @param iMatq matrix of indices for partial correlations equal to unique(royVarhelper(p)[, 1:2]).
//' @param iMate matrix of indices for partial correlations equal to royVarhelper(p, errors = T).
//' @param q number of unique partial correlations equal to choose(\eqn{p}, 2).
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
  arma::vec pcCovs(ncovs);

  int bw2 = pow(bw + 1, 2);
  IntegerVector bwinds = seq(0, bw2 - 1);
  int n2bw = ceil((N - 2*bw)/2);

  // Tapering weights
  NumericVector hu2s = cosTaper_cpp(seq(-bw, bw))[(seq(bw, 2*bw))-1];
  hu2s[0] = 1;

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

// [[Rcpp::export]]
double thetaHat_cpp(int i, int j, int l, int m, arma::mat ts, int n, NumericVector hu2s, arma::cube ccMat) {
  NumericVector vals1(n);
  NumericVector vals2(n);
  for(int h = 0; h < n; h++) {
    vals1(h) = ccMat(h, i, j);
    vals2(h) = ccMat(h, l, m);
  }
  return(sum(hu2s * vals1 * vals2));
}

// [[Rcpp::export]]
double deltaHat_cpp(int i, int j, int l, int m, arma::mat mvts, int n, NumericVector hu2s,
                    arma::vec ccs, arma::cube ccMat) {
  double crossProd = ccs(i)*ccs(j)*ccs(l)*ccs(m);
  return(thetaHat_cpp(i, j, l, m, mvts, n, hu2s, ccMat) / sqrt(crossProd));
}

//' @title Roy Asymptotic Variance
//'
//' @description This function calculates the asymptotic covariance matrix for correlations of a stationary multivariate time series as derived by Roy (1989).
//'
//' @param iMat Matrix of correlation indices
//' @param tsData Matrix of observed n-length p-variate time series
//' @param q Integer equal to the number of unique variables pairs given by choose(p, 2)
//' @param bw Bandwidth parameter
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
  IntegerVector mySeq = seq((-n+1), (n-1)) - 1;
  IntegerVector non0 = mySeq[abs(mySeq) <= bw];
  NumericVector hu2s = pow(cosTaper_cpp(non0), 2.0);
  int uLength = non0.size();

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
arma::mat royVar2_cpp(arma::mat iMat, arma::mat tsData, int q) {
  int N = iMat.n_rows;
  int n = tsData.n_rows;
  int p = tsData.n_cols;

  arma::mat pcMat = corrMat_cpp(tsData, false);

  // Creating hu2s for thetaHat_cpp
  IntegerVector mySeq = seq((-n+1), (n-1)) - 1;
  IntegerVector non0 = mySeq[abs(mySeq) <= 10];
  NumericVector hu2s = pow(cosTaper_cpp(non0), 2.0);
  int uLength = non0.size();

  // Calculating crossCov_cpp for lag 0
  arma::vec ccs(p);
  for(int i = 0; i < p; i++) {
    ccs(i) = crossCov_cpp(0, tsData.col(i), tsData.col(i));
  }

  // Calculating crossCov_cpp for different lags
  arma::cube ccMat(uLength, p, p);

  arma::vec royCov(N);
  for(int k = 0; k < N; k++) {
    int a = iMat(k, 0) - 1;
    int b = a + 1;
    int d = iMat(k, 2) - 1;
    int e = d + 1;

    for(int h = 0; h < uLength; h++) {
      ccMat(h, a, d) = crossCov_cpp(non0[h], tsData.col(a), tsData.col(d));
      ccMat(h, a, e) = crossCov_cpp(non0[h], tsData.col(a), tsData.col(e));
      ccMat(h, b, d) = crossCov_cpp(non0[h], tsData.col(b), tsData.col(d));
      ccMat(h, b, e) = crossCov_cpp(non0[h], tsData.col(b), tsData.col(e));
    }

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

//' Construct Block-Diagonal Matrix
//'
//' @param array3d 3D array of matrices to make into single block-diagonal matrix
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
arma::field<arma::mat> arrayEigen_cpp(arma::cube array3d) {

  // Extracting dimensions
  int dim3 = array3d.n_slices;

  arma::field<arma::mat> eigList(dim3*2);
  arma::vec eigval;
  arma::mat eigvec;

  for (int k = 0; k < dim3; k++) {
    assert(eig_sym(eigval, eigvec, array3d.slice(k)));
    eigList(k*2) = eigval;
    eigList(k*2+1) = eigvec;
  }

  return(eigList);
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
//' @description This function implements the a variance components model proposed by Fiecas et al. (2017).
//'
//' @param rs column vector containing q x K unique marginal or partial correlations.
//' @param sigmas 3D array of K estimated q x q covariance matrices for correlations.
//' @param sigEigs List of K matrices containing eigen decomposition matrices for covariance matrices contained in sigmas.
//' @param delta Threshold for algorithm
//' @param maxIters Maximum number of iterations for algorithm
//' @param sig0 Initial value for sigma parameter
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

  List resList = List::create(
    _["beta"] = beta,
    _["betaCov"] = betaCov,
    _["sigma"] = sigma,
    _["psi"] = psi
  );

  // Returning full or partial results to help with RAM issues
  if(smallRet) {
    List resList = List::create(
      _["beta"] = beta,
      _["betaCov"] = betaCov
    );
  } else {
    List resList = List::create(
      _["beta"] = beta,
      _["betaCov"] = betaCov,
      _["sigma"] = sigma,
      _["psi"] = psi
    );
  }

  return(resList);
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
  arma::field<arma::mat> eigDecomps = arrayEigen_cpp(sigmas);

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

