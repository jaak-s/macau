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

#include "macauvb.h"
#include "mvnormal.h"
#include "bpmfutils.h"

using namespace std; 
using namespace Eigen;

void MacauVB::addPrior(std::unique_ptr<ILatentPriorVB> & prior) {
  priors.push_back( std::move(prior) );
}

void MacauVB::setPrecision(double p) {
  alpha = p;
}

void MacauVB::setNiter(int ni) {
  niter = ni;
}

void MacauVB::setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  Y.resize(nrows, ncols);
  sparseFromIJV(Y, rows, cols, values, N);
  Yt = Y.transpose();
  mean_rating = Y.sum() / Y.nonZeros();
}

void MacauVB::setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  Ytest.resize(nrows, ncols);
  sparseFromIJV(Ytest, rows, cols, values, N);
}

void MacauVB::init() {
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  if (priors.size() != 2) {
    throw std::runtime_error("Only 2 priors are supported.");
  }
  init_bmrng(seed1);
  // means
  MatrixXd* U = new MatrixXd(num_latent, Y.rows());
  MatrixXd* V = new MatrixXd(num_latent, Y.cols());
  bmrandn(*U);
  bmrandn(*V);
  (*U) *= latent_init_std;
  (*V) *= latent_init_std;
  samples_mean.push_back( std::move(std::unique_ptr<MatrixXd>(U)) );
  samples_mean.push_back( std::move(std::unique_ptr<MatrixXd>(V)) );

  // vars
  double init_var = 4.0;
  MatrixXd* Uvar = new MatrixXd(num_latent, Y.rows());
  MatrixXd* Vvar = new MatrixXd(num_latent, Y.cols());
  (*Uvar).setConstant(init_var);
  (*Vvar).setConstant(init_var);
  samples_var.push_back( std::move(std::unique_ptr<MatrixXd>(Uvar)) );
  samples_var.push_back( std::move(std::unique_ptr<MatrixXd>(Vvar)) );
}

void MacauVB::run() {
  init();
  if (verbose) {
    std::cout << "Sampling" << endl;
  }

  const int num_rows = Y.rows();
  const int num_cols = Y.cols();
  predictions     = VectorXd::Zero( Ytest.nonZeros() );

  auto start = tick();
  for (int i = 0; i < niter; i++) {
    auto starti = tick();

    // update latent vectors
    priors[0]->update_latents(*samples_mean[0], *samples_var[0], Yt, mean_rating,
                              *samples_mean[1], *samples_var[1], alpha);
    priors[1]->update_latents(*samples_mean[1], *samples_var[1],  Y, mean_rating,
                              *samples_mean[0], *samples_var[0], alpha);

    // update hyperparams
    priors[0]->update_prior(*samples_mean[0], *samples_var[0]);
    priors[1]->update_prior(*samples_mean[1], *samples_var[1]);

    auto endi = tick();
    auto elapsed = endi - start;
    double updates_per_sec = (i + 1) * (num_rows + num_cols) / elapsed;
    double elapsedi = endi - starti;

    if (verbose) {
      printStatus(i, NAN, elapsedi, updates_per_sec);
    }
    //rmse_test = eval.second;
  }
}

void MacauVB::printStatus(int i, double rmse, double elapsedi, double updates_per_sec) {
  printf("Iter %d: RMSE: %4.4f\tFU(%1.2e) FV(%1.2e) [took %0.1fs, Updates/sec: %6.1f]\n", i, rmse, samples_mean[0]->norm(), samples_mean[1]->norm(), elapsedi, updates_per_sec);
  /*
  double norm0 = priors[0]->getLinkNorm();
  double norm1 = priors[1]->getLinkNorm();
  if (!std::isnan(norm0) || !std::isnan(norm1)) {
    printf("          [Side info] ");
    if (!std::isnan(norm0)) printf("U.link(%1.2e) U.lambda(%.1f) ", norm0, priors[0]->getLinkLambda());
    if (!std::isnan(norm1)) printf("V.link(%1.2e) V.lambda(%.1f)",   norm1, priors[1]->getLinkLambda());
    printf("\n");
  }*/
}
