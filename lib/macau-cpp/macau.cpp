#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>

#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>

#include <omp.h>

#include "macau.h"
#include "mvnormal.h"
#include "bpmfutils.h"
#include "latentprior.h"

using namespace std; 
using namespace Eigen;

void Macau::addPrior(unique_ptr<ILatentPrior> & prior) {
  priors.push_back( std::move(prior) );
}

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
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  if (priors.size() != 2) {
    throw std::runtime_error("Only 2 priors are supported.");
  }
  init_bmrng(seed1);
  MatrixXd* U = new MatrixXd(num_latent, Y.rows());
  MatrixXd* V = new MatrixXd(num_latent, Y.cols());
  U->setZero();
  V->setZero();
  samples.push_back( std::move(std::unique_ptr<MatrixXd>(U)) );
  samples.push_back( std::move(std::unique_ptr<MatrixXd>(V)) );
}

Macau::~Macau() {
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
    if (verbose && i == burnin) {
      printf(" ====== Burn-in complete, averaging samples ====== \n");
    }
    auto starti = tick();

    // sample latent vectors
    priors[0]->sample_latents(*samples[0], Yt, mean_rating, *samples[1], alpha, num_latent);
    priors[1]->sample_latents(*samples[1], Y,  mean_rating, *samples[0], alpha, num_latent);

    // Sample hyperparams
    priors[0]->update_prior(*samples[0]);
    priors[1]->update_prior(*samples[1]);

    auto eval = eval_rmse(Ytest, (i < burnin) ? 0 : (i - burnin + 1), predictions, *samples[1], *samples[0], mean_rating);

    auto endi = tick();
    auto elapsed = endi - start;
    double samples_per_sec = (i + 1) * (num_rows + num_cols) / elapsed;
    double elapsedi = endi - starti;

    if (verbose) {
      printStatus(i, eval.first, eval.second, elapsedi, samples_per_sec);
    }
    rmse_test = eval.second;
  }
}

void Macau::printStatus(int i, double rmse, double rmse_avg, double elapsedi, double samples_per_sec) {
  printf("Iter %d: RMSE: %4.4f\tavg RMSE: %4.4f  FU(%1.2e) FV(%1.2e) [took %0.1fs, Samples/sec: %6.1f]\n", i, rmse, rmse_avg, samples[0]->norm(), samples[1]->norm(), elapsedi, samples_per_sec);
  double norm0 = priors[0]->getLinkNorm();
  double norm1 = priors[1]->getLinkNorm();
  if (!std::isnan(norm0) || !std::isnan(norm1)) {
    printf("          [Side info] ");
    if (!std::isnan(norm0)) printf("U.link(%1.2e) U.lambda(%.1f) ", norm0, priors[0]->getLinkLambda());
    if (!std::isnan(norm1)) printf("V.link(%1.2e) V.lambda(%.1f)",   norm1, priors[1]->getLinkLambda());
    printf("\n");
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
