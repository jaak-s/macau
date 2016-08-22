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
  mean_value = Y.sum() / Y.nonZeros();
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
    std::cout << "Estimating model" << endl;
  }

  const int num_rows = Y.rows();
  const int num_cols = Y.cols();
  predictions     = VectorXd::Zero( Ytest.nonZeros() );

  auto start = tick();
  /*
  std::cout << "Umean: " << *samples_mean[0] << std::endl;
  std::cout << "Uvar:  " << *samples_var[0]  << std::endl;
  std::cout << "Vmean: " << *samples_mean[1] << std::endl;
  std::cout << "Vvar:  " << *samples_var[1]  << std::endl;
  std::cout << "Uprior.mu_mean:  " << ((BPMFPriorVB*)priors[0].get())->mu_mean   << std::endl;
  std::cout << "Uprior.mu_var:   " << ((BPMFPriorVB*)priors[0].get())->mu_var    << std::endl;
  std::cout << "Uprior.lambda_b: " << ((BPMFPriorVB*)priors[0].get())->lambda_b  << std::endl;
  std::cout << "Uprior.lambda_a0:" << ((BPMFPriorVB*)priors[0].get())->lambda_a0 << std::endl;
  std::cout << "Uprior.lambda_b0:" << ((BPMFPriorVB*)priors[0].get())->lambda_b0 << std::endl;
  std::cout << "Uprior.b0:       " << ((BPMFPriorVB*)priors[0].get())->b0        << std::endl;
  std::cout << "Uprior.Elambda:  " << ((BPMFPriorVB*)priors[0].get())->getElambda(Y.rows()) << std::endl;
  std::cout << "-----\n";
  */
  for (int i = 0; i < niter; i++) {
    auto starti = tick();

    // update latent vectors
    priors[0]->update_latents(*samples_mean[0], *samples_var[0], Yt, mean_value,
                              *samples_mean[1], *samples_var[1], alpha);
    //std::cout << "Umean: " << *samples_mean[0] << std::endl;
    //std::cout << "Uvar:  " << *samples_var[0]  << std::endl;
    priors[1]->update_latents(*samples_mean[1], *samples_var[1],  Y, mean_value,
                              *samples_mean[0], *samples_var[0], alpha);
    //std::cout << "Vmean: " << *samples_mean[1] << std::endl;
    //std::cout << "Vvar:  " << *samples_var[1]  << std::endl;

    // update hyperparams
    priors[0]->update_prior(*samples_mean[0], *samples_var[0]);
    priors[1]->update_prior(*samples_mean[1], *samples_var[1]);

    rmse_test = eval_rmse(predictions, Ytest, *samples_mean[1], *samples_mean[0], mean_value);

    auto endi = tick();
    auto elapsed = endi - start;
    double updates_per_sec = (i + 1) * (num_rows + num_cols) / elapsed;
    double elapsedi = endi - starti;

    if (verbose) {
      printStatus(i, rmse_test, elapsedi, updates_per_sec);
    }
  }
  if (save_model) {
    saveModel();
  }
}

void MacauVB::printStatus(int i, double rmse, double elapsedi, double updates_per_sec) {
  double norm0 = priors[0]->getLinkNorm();
  double norm1 = priors[1]->getLinkNorm();
  printf("Iter %3d: RMSE: %4.4f\tU:[%1.2e, %1.2e]  Side:[%1.2e, %1.2e]  [took %0.1fs]\n", i, rmse, samples_mean[0]->norm(), samples_mean[1]->norm(), norm0, norm1, elapsedi);
}

double inline sqr(double x) {
  return x * x;
}

double eval_rmse(VectorXd & predictions, const SparseMatrix<double> & P, const MatrixXd &sample_m, const MatrixXd &sample_u, const double mean_value)
{
  double se = 0.0;
#pragma omp parallel for schedule(dynamic,8) reduction(+:se)
  for (int k = 0; k < P.outerSize(); ++k) {
    int idx = P.outerIndexPtr()[k];
    for (SparseMatrix<double>::InnerIterator it(P,k); it; ++it) {
      const double pred = sample_m.col(it.col()).dot(sample_u.col(it.row())) + mean_value;
      se += sqr(it.value() - pred);
      predictions[idx++] = pred;
    }
  }

  return sqrt( se / P.nonZeros() );
}

Eigen::VectorXd MacauVB::getStds() {
  VectorXd stds(Ytest.nonZeros());
  const MatrixXd & Umean = *samples_mean[0];
  const MatrixXd & Vmean = *samples_mean[1];
  const MatrixXd & Uvar  = *samples_var[0];
  const MatrixXd & Vvar  = *samples_var[1];
#pragma omp parallel for schedule(dynamic, 4)
  for (int col = 0; col < Ytest.outerSize(); col++) {
    int idx = Ytest.outerIndexPtr()[col];
    for (SparseMatrix<double>::InnerIterator it(Ytest, col); it; ++it) {
      int row = it.row();
      double var = 0.0;
      for (int d = 0; d < num_latent; d++) {
        var += Umean(d, row) * Umean(d, row) * Vvar(d, col) +
               Vmean(d, col) * Vmean(d, col) * Uvar(d, row) +
               Uvar(d, row) * Vvar(d, col);
      }
      stds(idx++) = sqrt(var);
    }
  }
  return stds;
}

Eigen::MatrixXd MacauVB::getTestData() {
  MatrixXd coords(Ytest.nonZeros(), 3);
#pragma omp parallel for schedule(static)
  for (int k = 0; k < Ytest.outerSize(); ++k) {
    int idx = Ytest.outerIndexPtr()[k];
    for (SparseMatrix<double>::InnerIterator it(Ytest,k); it; ++it) {
      coords(idx, 0) = it.row();
      coords(idx, 1) = it.col();
      coords(idx, 2) = it.value();
      idx++;
    }
  }
  return coords;
}

void MacauVB::saveModel() {
  // saving latent matrices
  for (unsigned int i = 0; i < samples_mean.size(); i++) {
    writeToCSVfile(save_prefix + "-U" + std::to_string(i+1) + "-latents-mean.csv", *samples_mean[i]);
    writeToCSVfile(save_prefix + "-U" + std::to_string(i+1) + "-latents-var.csv", *samples_var[i]);
    priors[i]->saveModel(save_prefix + "-U" + std::to_string(i+1));
  }

  // saving global mean
  VectorXd means(1);
  means << mean_value;
  writeToCSVfile(save_prefix + "-meanvalue.csv", means);
}
