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
#include "latentprior.h"

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
  mean_rating = Y.sum() / Y.nonZeros();
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
  sample_u.resize(num_latent, Y.rows());
  sample_m.resize(num_latent, Y.cols());
  sample_u.setZero(); 
  sample_m.setZero(); 
}

inline double sqr(double x) { return x*x; }

void Macau::run() {
  init();
  std::cout << "Sampling" << endl;

  const int num_rows = Y.rows();
  const int num_cols = Y.cols();
  VectorXd predictions = VectorXd::Zero( Ytest.nonZeros() );

  auto start = tick();
  for (int i = 0; i < burnin + nsamples; i++) {
    if (i == burnin) {
      printf(" ====== Burn-in complete, averaging samples ====== \n");
    }
    auto starti = tick();

    // sample latent vectors
    prior_u.sample_latents(sample_u, Yt, mean_rating, sample_m, alpha, num_latent);
    prior_m.sample_latents(sample_m, Y,  mean_rating, sample_u, alpha, num_latent);

    // Sample hyperparams
    prior_u.update_prior(sample_u);
    prior_m.update_prior(sample_m);

    auto eval = eval_rmse(Ytest, (i < burnin) ? 0 : (i - burnin + 1), predictions, sample_m, sample_u, mean_rating);

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
