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

void Macau::setPrecision(double p) {
  alpha = p;
}

void Macau::setSamples(int b, int n) {
  burnin = b;
  nsamples = n;
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

double Macau::getRmseTest() { return rmse_test; }

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
  std::cout << "Sampling" << endl;

  const int num_rows = Y.rows();
  const int num_cols = Y.cols();
  printf("Size of Ytest: %d\n", Ytest.nonZeros());
  VectorXd predictions = VectorXd::Zero( Ytest.nonZeros() );

  auto start = tick();
  for (int i = 0; i < burnin + nsamples; i++) {
    if (i == burnin) {
      printf(" ====== Burn-in complete, averaging samples ====== \n");
    }
    auto starti = tick();

#pragma omp parallel for
    for(int uu = 0; uu < num_cols; ++uu) {
      sample_latent(sample_u, uu, Yt, mean_rating, sample_m, alpha, mu_u, Lambda_u, num_latent);
    }

#pragma omp parallel for
    for(int mm = 0; mm < num_rows; ++mm) {
      sample_latent(sample_m, mm, Y, mean_rating, sample_u, alpha, mu_m, Lambda_m, num_latent);
    }

    // Sample hyperparams
    tie(mu_u, Lambda_u) = CondNormalWishart(sample_u, mu0_u, b0_u, WI_u, df_u);
    tie(mu_m, Lambda_m) = CondNormalWishart(sample_m, mu0_m, b0_m, WI_m, df_m);

    auto eval = eval_rmse(Ytest, (i < burnin) ? 0 : (i - burnin), predictions, sample_m, sample_u, mean_rating);

    auto endi = tick();
    auto elapsed = endi - start;
    double samples_per_sec = (i + 1) * (num_rows + num_cols) / elapsed;
    double elapsedi = endi - starti;

    printf("Iter %d: RMSE: %4.4f\tavg RMSE: %4.4f\tFU(%1.3e) FV(%1.3e) [took %0.1fs, Samples/sec: %6.1f]\n", i, eval.first, eval.second, sample_u.norm(), sample_m.norm(), elapsedi, samples_per_sec);
    rmse_test = eval.second;
  }
}

std::pair<double,double> eval_rmse(SparseMatrix<double> & P, const int n, VectorXd & predictions, const MatrixXd &sample_m, const MatrixXd &sample_u, double mean_rating)
{
  // TODO: parallelize
  double se = 0.0, se_avg = 0.0;
  unsigned idx = 0;
  for (int k = 0; k < P.outerSize(); ++k) {
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
  MatrixXd MM = MatrixXd::Zero(num_latent, num_latent);
  VectorXd rr = VectorXd::Zero(num_latent);
  for (SparseMatrix<double>::InnerIterator it(mat, mm); it; ++it) {
    auto col = samples.col(it.row());
    MM.noalias() += col * col.transpose();
    rr.noalias() += col * ((it.value() - mean_rating) * alpha);
  }

  Eigen::LLT<MatrixXd> chol = (Lambda_u + alpha * MM).llt();
  if(chol.info() != Eigen::Success) {
    std::cout << "MM:\n" << MM << std::endl;
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
