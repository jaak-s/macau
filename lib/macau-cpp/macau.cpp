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
#include <signal.h>

#include "macau.h"
#include "mvnormal.h"
#include "bpmfutils.h"
#include "latentprior.h"

using namespace std; 
using namespace Eigen;

static volatile bool keepRunning = true;

void intHandler(int dummy) {
  keepRunning = false;
  printf("[Received Ctrl-C. Stopping after finishing the current iteration.]\n");
}

void Macau::addPrior(unique_ptr<ILatentPrior> & prior) {
  priors.push_back( std::move(prior) );
}

void Macau::setPrecision(double p) {
  noise.reset(new FixedGaussianNoise(p));
}

void Macau::setAdaptivePrecision(double sn_init, double sn_max) {
  noise.reset(new AdaptiveGaussianNoise(sn_init, sn_max));
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
  noise->init(Y, mean_rating);
  keepRunning = true;
}

Macau::~Macau() {
}

inline double sqr(double x) { return x*x; }

void Macau::run() {
  init();
  if (verbose) {
    std::cout << noise->getInitStatus() << endl;
    std::cout << "Sampling" << endl;
  }
  if (save_model) {
    saveGlobalParams();
  }
  signal(SIGINT, intHandler);

  const int num_rows = Y.rows();
  const int num_cols = Y.cols();
  predictions     = VectorXd::Zero( Ytest.nonZeros() );
  predictions_var = VectorXd::Zero( Ytest.nonZeros() );

  auto start = tick();
  for (int i = 0; i < burnin + nsamples; i++) {
    if (keepRunning == false) {
      keepRunning = true;
      break;
    }
    if (verbose && i == burnin) {
      printf(" ====== Burn-in complete, averaging samples ====== \n");
    }
    auto starti = tick();

    // sample latent vectors
    noise->sample_latents(priors[0], *samples[0], Yt, mean_rating, *samples[1], num_latent);
    noise->sample_latents(priors[1], *samples[1], Y,  mean_rating, *samples[0], num_latent);

    // Sample hyperparams
    priors[0]->update_prior(*samples[0]);
    priors[1]->update_prior(*samples[1]);

    noise->update(Y, mean_rating, samples);

    auto eval = eval_rmse(Ytest, (i < burnin) ? 0 : (i - burnin), predictions, predictions_var, *samples[1], *samples[0], mean_rating);

    auto endi = tick();
    auto elapsed = endi - start;
    double samples_per_sec = (i + 1) * (num_rows + num_cols) / elapsed;
    double elapsedi = endi - starti;

    if (save_model && i >= burnin) {
      saveModel(i - burnin + 1);
    }
    if (verbose) {
      printStatus(i, eval.first, eval.second, elapsedi, samples_per_sec);
    }
    rmse_test = eval.second;
  }
}

void Macau::printStatus(int i, double rmse, double rmse_avg, double elapsedi, double samples_per_sec) {
  double norm0 = priors[0]->getLinkNorm();
  double norm1 = priors[1]->getLinkNorm();
  printf("Iter %3d: RMSE: %.4f (1samp: %.4f)  U:[%1.2e, %1.2e]  Side:[%1.2e, %1.2e] %s [took %0.1fs]\n", i, rmse_avg, rmse, samples[0]->norm(), samples[1]->norm(), norm0, norm1, noise->getStatus().c_str(), elapsedi);
  // if (!std::isnan(norm0)) printf("U.link(%1.2e) U.lambda(%.1f) ", norm0, priors[0]->getLinkLambda());
  // if (!std::isnan(norm1)) printf("V.link(%1.2e) V.lambda(%.1f)",   norm1, priors[1]->getLinkLambda());
}

Eigen::VectorXd Macau::getStds() {
  VectorXd std(Ytest.nonZeros());
  if (nsamples <= 1) {
    std.setConstant(NAN);
    return std;
  }
  const int n = std.size();
  const double inorm = 1.0 / (nsamples - 1);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    std[i] = sqrt(predictions_var[i] * inorm);
  }
  return std;
}

// assumes matrix (not tensor)
Eigen::MatrixXd Macau::getTestData() {
  MatrixXd coords(Ytest.nonZeros(), 3);
#pragma omp parallel for schedule(dynamic, 2)
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

void Macau::saveModel(int isample) {
  string fprefix = save_prefix + "-sample" + std::to_string(isample) + "-";
  // saving latent matrices
  for (unsigned int i = 0; i < samples.size(); i++) {
    writeToCSVfile(fprefix + "U" + std::to_string(i+1) + "-latents.csv", *samples[i]);
    priors[i]->saveModel(fprefix + "U" + std::to_string(i+1));
  }
}

void Macau::saveGlobalParams() {
  VectorXd means(1);
  means << mean_rating;
  writeToCSVfile(save_prefix + "-meanvalue.csv", means);
}
