#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>                                                                                                         
#include <fstream>
#include <string>
#include <algorithm>
#include <random>

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>

#include <omp.h>

#include "macau.h"
#include "mvnormal.h"
#include "utils.h"

using namespace std; 
using namespace Eigen;

Macau::Macau() {
  num_latent = 10;
}

Macau::Macau(int num_latent_v) {
  num_latent = num_latent_v;
}

void Macau::setPrecision(double p) {
  alpha = p;
}

void Macau::setSamples(int b, int n) {
  burnin = b;
  nsamples = n;
}

void Macau::sum(double* X, double* Y, double* S, int N) {
#pragma omp parallel for schedule(static) 
  for (int n = 0; n < N; n++) {
    S[n] = X[n] + Y[n];
  }
}

void Macau::setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  Y.resize(nrows, ncols);
  sparseFromIJV(Y, rows, cols, values, N);
  Yt = Y.transpose();
}

void Macau::setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  Ytest.resize(nrows, ncols);
  sparseFromIJV(Ytest, rows, cols, values, N);
}

void sparseFromIJV(SparseMatrix<double> &X, int* rows, int* cols, double* values, int N) {
  typedef Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(N);
  for (int n = 0; n < N; n++) {
    tripletList.push_back(T(rows[n], cols[n], values[n]));
  }
  X.setFromTriplets(tripletList.begin(), tripletList.end());
}

void Macau::init() {
  mu_u.resize(num_latent);
  mu_m.resize(num_latent);
  mu_u.setZero();
  mu_m.setZero();

  Lambda_u.resize(num_latent, num_latent);
  Lambda_m.resize(num_latent, num_latent);
  Lambda_u.setIdentity();
  Lambda_m.setIdentity();
  Lambda_u *= 10;
  Lambda_m *= 10;

  sample_u.resize(num_latent, Y.rows());
  sample_m.resize(num_latent, Y.cols());
  sample_u.setZero(); 
  sample_m.setZero(); 

  // parameters of Inv-Whishart distribution
  WI_u.resize(num_latent, num_latent);
  WI_u.setIdentity();
  mu0_u.resize(num_latent);
  mu0_u.setZero();

  WI_m.resize(num_latent, num_latent);
  WI_m.setIdentity();
  mu0_m.resize(num_latent);
  mu0_m.setZero();

}

inline double sqr(double x) { return x*x; }


void Macau::run() {
  init();
}

std::pair<double,double> eval_rmse(SparseMatrix<double> & P, int n, VectorXd & predictions, const MatrixXd &sample_m, const MatrixXd &sample_u, double mean_rating)
{
  // TODO: parallelize
  double se = 0.0, se_avg = 0.0;
  unsigned idx = 0;
  for (int k=0; k<P.outerSize(); ++k) {
    for (SparseMatrix<double>::InnerIterator it(P,k); it; ++it) {
      const double pred = sample_m.col(it.col()).dot(sample_u.col(it.row())) + mean_rating;
      se += sqr(it.value() - pred);

      const double pred_avg = (n == 0) ? pred : (predictions[idx] + (pred - predictions[idx]) / n);
      se_avg += sqr(it.value() - pred_avg);
      predictions[idx++] = pred_avg;
    }
  }

  const unsigned N = P.nonZeros();
  const double rmse = sqrt( se / N );
  const double rmse_avg = sqrt( se_avg / N );
  return std::make_pair(rmse, rmse_avg);
}

void sample_latent(MatrixXd &s, int mm, const SparseMatrix<double> &mat, double mean_rating,
    const MatrixXd &samples, double alpha, const VectorXd &mu_u, const MatrixXd &Lambda_u,
    const int num_latent)
{
  // TODO: add cholesky update version
  MatrixXd MM(num_latent, num_latent);
  VectorXd rr(num_latent);
  for (SparseMatrix<double>::InnerIterator it(mat, mm); it; ++it) {
    auto col = samples.col(it.row());
    MM.noalias() += col * col.transpose();
    rr.noalias() += col * ((it.value() - mean_rating) * alpha);
  }

  Eigen::LLT<MatrixXd> chol = (Lambda_u + alpha * MM).llt();
  if(chol.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky Decomposition failed!");
  }

  rr.noalias() += Lambda_u * mu_u;
  chol.matrixL().solveInPlace(rr);
  for (int i = 0; i < num_latent; i++) {
    rr[i] += randn0();
  }
  chol.matrixU().solveInPlace(rr);
  s.col(mm).noalias() = rr;
}
