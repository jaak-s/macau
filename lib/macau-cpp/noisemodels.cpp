#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <cmath>

#include "noisemodels.h"

using namespace Eigen;

template<class T>
void INoiseModelDisp<T>::sample_latents(std::unique_ptr<ILatentPrior> & prior,
                                Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                double mean_value, const Eigen::MatrixXd &samples, const int num_latent)
{
  prior->sample_latents(static_cast<T *>(this), U, mat, mean_value, samples, num_latent);
}


////  AdaptiveGaussianNoise  ////
void AdaptiveGaussianNoise::init(const Eigen::SparseMatrix<double> &train, double mean_value) {
  double se = 0.0;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
  for (int k = 0; k < train.outerSize(); ++k) {
    for (SparseMatrix<double>::InnerIterator it(train,k); it; ++it) {
      se += square(it.value() - mean_value);
    }
  }

  var_total = se / train.nonZeros();
  if (var_total <= 0.0 || std::isnan(var_total)) {
    // if var cannot be computed using 1.0
    var_total = 1.0;
  }
  // Var(noise) = Var(total) / (SN + 1)
  alpha     = (sn_init + 1.0) / var_total;
  alpha_max = (sn_max + 1.0) / var_total;
}

void AdaptiveGaussianNoise::update(const Eigen::SparseMatrix<double> &train, double mean_value, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples)
{
  double sumsq = 0.0;
  MatrixXd & U = *samples[0];
  MatrixXd & V = *samples[1];

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
  for (int j = 0; j < train.outerSize(); j++) {
    auto Vj = V.col(j);
    for (SparseMatrix<double>::InnerIterator it(train, j); it; ++it) {
      double Yhat = Vj.dot( U.col(it.row()) ) + mean_value;
      sumsq += square(Yhat - it.value());
    }
  }
  // (a0, b0) correspond to a prior of 1 sample of noise with full variance
  double a0 = 0.5;
  double b0 = 0.5 * var_total;
  double aN = a0 + train.nonZeros() / 2.0;
  double bN = b0 + sumsq / 2.0;
  alpha = rgamma(aN, 1.0 / bN);
  if (alpha > alpha_max) {
    alpha = alpha_max;
  }
}


template class INoiseModelDisp<FixedGaussianNoise>;
template class INoiseModelDisp<AdaptiveGaussianNoise>;
template class INoiseModelDisp<ProbitNoise>;
