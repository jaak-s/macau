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
#include "sparsetensor.h"

using namespace std; 
using namespace Eigen;

static volatile bool keepRunning = true;

void intHandler(int dummy) {
  keepRunning = false;
  printf("[Received Ctrl-C. Stopping after finishing the current iteration.]\n");
}

template<class DType, class NType> 
void MacauX<DType, NType>::addPrior(unique_ptr<ILatentPrior> & prior) {
  priors.push_back( std::move(prior) );
}

template<class DType, class NType> 
void MacauX<DType, NType>::setSamples(int b, int n) {
  burnin = b;
  nsamples = n;
}

template<class DType, class NType> 
void MacauX<DType, NType>::setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  data.setTrain(rows, cols, values, N, nrows, ncols);
}

template<class DType, class NType> 
void MacauX<DType, NType>::setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  data.setTest(rows, cols, values, N, nrows, ncols);
}

template<class DType, class NType> 
void MacauX<DType, NType>::setRelationData(int** idx, int nmodes, double* values, int nnz, int* dims) {
  data.setTrain(idx, nmodes, values, nnz, dims);
}

template<class DType, class NType> 
void MacauX<DType, NType>::setRelationDataTest(int** idx, int nmodes, double* values, int nnz, int* dims) {
  data.setTest(idx, nmodes, values, nnz, dims);
}

template<class DType, class NType> 
double MacauX<DType, NType>::getRmseTest() { return rmse_test; }

template<class DType, class NType> 
void MacauX<DType, NType>::init() {
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  if (priors.size() != 2) {
    throw std::runtime_error("Only 2 priors are supported.");
  }
  init_bmrng(seed1);
  VectorXi dims = data.getDims();
  for (int mode = 0; mode < dims.size(); mode++) {
    MatrixXd* U = new MatrixXd(num_latent, dims(mode));
    U->setZero();
    samples.push_back( std::move(std::unique_ptr<MatrixXd>(U)) );
  }
  noise.init(data);
  keepRunning = true;
}

Macau::~Macau() {
}

inline double sqr(double x) { return x*x; }

template<class DType, class NType> 
void MacauX<DType, NType>::run() {
  init();
  if (verbose) {
    std::cout << noise.getInitStatus() << endl;
    std::cout << "Sampling" << endl;
  }
  if (save_model) {
    saveGlobalParams();
  }
  signal(SIGINT, intHandler);

  const VectorXi dims = data.getDims();
  predictions     = VectorXd::Zero( data.getTestNonzeros() );
  predictions_var = VectorXd::Zero( data.getTestNonzeros() );

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
    for (int mode = 0; mode < dims.size(); mode++) {
      priors[mode]->sample_latents(noise, data, samples, mode, num_latent);
    }

    // Sample hyperparams
    for (int mode = 0; mode < dims.size(); mode++) {
      priors[mode]->update_prior(*samples[mode]);
    }

    noise.update(data, samples);

    noise.evalModel(data, (i < burnin) ? 0 : (i - burnin), predictions, predictions_var, samples);
    

    auto endi = tick();
    auto elapsed = endi - start;
    double samples_per_sec = (i + 1) * (dims.sum()) / elapsed;
    double elapsedi = endi - starti;

    if (save_model && i >= burnin) {
      saveModel(i - burnin + 1);
    }
    if (verbose) {
      printStatus(i, elapsedi, samples_per_sec);
    }
    rmse_test = noise.getEvalMetric();
  }
}

template<class DType, class NType> 
void MacauX<DType, NType>::printStatus(int i, double elapsedi, double samples_per_sec) {
  double norm0 = priors[0]->getLinkNorm();
  double norm1 = priors[1]->getLinkNorm();
  printf("Iter %3d: %s  U:[%1.2e, %1.2e]  Side:[%1.2e, %1.2e] %s [took %0.1fs]\n", i, noise.getEvalString().c_str(), samples[0]->norm(), samples[1]->norm(), norm0, norm1, noise.getStatus().c_str(), elapsedi);
  // if (!std::isnan(norm0)) printf("U.link(%1.2e) U.lambda(%.1f) ", norm0, priors[0]->getLinkLambda());
  // if (!std::isnan(norm1)) printf("V.link(%1.2e) V.lambda(%.1f)",   norm1, priors[1]->getLinkLambda());
}

template<class DType, class NType> 
Eigen::VectorXd MacauX<DType, NType>::getStds() {
  VectorXd std( data.getTestNonzeros() );
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

template<class DType, class NType> 
void MacauX<DType, NType>::saveModel(int isample) {
  string fprefix = save_prefix + "-sample" + std::to_string(isample) + "-";
  // saving latent matrices
  for (unsigned int i = 0; i < samples.size(); i++) {
    writeToCSVfile(fprefix + "U" + std::to_string(i+1) + "-latents.csv", *samples[i]);
    priors[i]->saveModel(fprefix + "U" + std::to_string(i+1));
  }
}

template<class DType, class NType> 
void MacauX<DType, NType>::saveGlobalParams() {
  VectorXd means(1);
  means << data.getMeanValue();
  writeToCSVfile(save_prefix + "-meanvalue.csv", means);
}

Macau* make_macau_probit(int nmodes, int num_latent) {
  switch (nmodes) {
    case 2: return new MacauX<MatrixData, ProbitNoise>(num_latent);
    case 3: return new MacauX<TensorData<3>, ProbitNoise>(num_latent);
    case 4: return new MacauX<TensorData<4>, ProbitNoise>(num_latent);
    case 5: return new MacauX<TensorData<5>, ProbitNoise>(num_latent);
    case 6: return new MacauX<TensorData<6>, ProbitNoise>(num_latent);
  }
  throw std::runtime_error("Unsupported value for nmodes: allowed values are 2 to 6.");
}

Macau* make_macau_fixed(int nmodes, int num_latent, double precision) {
  FixedGaussianNoise fnoise(precision);
  switch (nmodes) {
    case 2: return new MacauX<MatrixData, FixedGaussianNoise>(num_latent, fnoise);
    case 3: return new MacauX<TensorData<3>, FixedGaussianNoise>(num_latent, fnoise);
    case 4: return new MacauX<TensorData<4>, FixedGaussianNoise>(num_latent, fnoise);
    case 5: return new MacauX<TensorData<5>, FixedGaussianNoise>(num_latent, fnoise);
    case 6: return new MacauX<TensorData<6>, FixedGaussianNoise>(num_latent, fnoise);
  }
  throw std::runtime_error("Unsupported value for nmodes: allowed values are 2 to 6.");
}

Macau* make_macau_adaptive(int nmodes, int num_latent, double sn_init, double sn_max) {
  AdaptiveGaussianNoise anoise(sn_init, sn_max);
  switch (nmodes) {
    case 2: return new MacauX<MatrixData,    AdaptiveGaussianNoise>(num_latent, anoise);
    case 3: return new MacauX<TensorData<3>, AdaptiveGaussianNoise>(num_latent, anoise);
    case 4: return new MacauX<TensorData<4>, AdaptiveGaussianNoise>(num_latent, anoise);
    case 5: return new MacauX<TensorData<5>, AdaptiveGaussianNoise>(num_latent, anoise);
    case 6: return new MacauX<TensorData<6>, AdaptiveGaussianNoise>(num_latent, anoise);
  }
  throw std::runtime_error("Unsupported value for nmodes: allowed values are 2 to 6.");
}

template class MacauX<MatrixData, FixedGaussianNoise>;
template class MacauX<MatrixData, AdaptiveGaussianNoise>;
template class MacauX<MatrixData, ProbitNoise>;

template class MacauX<TensorData<3>, FixedGaussianNoise>;
template class MacauX<TensorData<3>, AdaptiveGaussianNoise>;
template class MacauX<TensorData<3>, ProbitNoise>;

template class MacauX<TensorData<4>, FixedGaussianNoise>;
template class MacauX<TensorData<4>, AdaptiveGaussianNoise>;
template class MacauX<TensorData<4>, ProbitNoise>;

template class MacauX<TensorData<5>, FixedGaussianNoise>;
template class MacauX<TensorData<5>, AdaptiveGaussianNoise>;
template class MacauX<TensorData<5>, ProbitNoise>;

template class MacauX<TensorData<6>, FixedGaussianNoise>;
template class MacauX<TensorData<6>, AdaptiveGaussianNoise>;
template class MacauX<TensorData<6>, ProbitNoise>;
