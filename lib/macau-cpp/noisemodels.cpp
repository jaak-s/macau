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

template<class T>
void INoiseModelDisp<T>::sample_latents(std::unique_ptr<ILatentPrior> & prior,
                                        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                                        std::unique_ptr<IData> & data,
                                        int mode,
                                        const int num_latent)
{
  data->sample_latents(prior, static_cast<T *>(this), samples, mode, num_latent);
}

template<class T>
void INoiseModelDisp<T>::init(std::unique_ptr<IData> & data)
{
  data->initNoise(static_cast<T *>(this));
}

template<class T>
void INoiseModelDisp<T>::update(std::unique_ptr<IData> & data, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) {
  data->updateNoise(static_cast<T *>(this), samples);
}

template<class T>
void INoiseModelDisp<T>::evalModel(std::unique_ptr<IData> & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) {
  data->evalModel(static_cast<T *>(this), n, predictions, predictions_var, samples);
}


////  AdaptiveGaussianNoise  ////
void init_noise(MatrixData* matrixData, AdaptiveGaussianNoise* noise) {
//void AdaptiveGaussianNoise::init(const Eigen::SparseMatrix<double> &train, double mean_value) {
  double se = 0.0;
  double mean_value = matrixData->mean_value;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
  for (int k = 0; k < matrixData->Y.outerSize(); ++k) {
    for (SparseMatrix<double>::InnerIterator it(matrixData->Y, k); it; ++it) {
      se += square(it.value() - mean_value);
    }
  }

  noise->var_total = se / matrixData->Y.nonZeros();
  if (noise->var_total <= 0.0 || std::isnan(noise->var_total)) {
    // if var cannot be computed using 1.0
    noise->var_total = 1.0;
  }
  // Var(noise) = Var(total) / (SN + 1)
  noise->alpha     = (noise->sn_init + 1.0) / noise->var_total;
  noise->alpha_max = (noise->sn_max + 1.0)  / noise->var_total;
}

void update_noise(MatrixData* data, AdaptiveGaussianNoise* noise, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples)
{
  double sumsq = 0.0;
  MatrixXd & U = *samples[0];
  MatrixXd & V = *samples[1];

  Eigen::SparseMatrix<double> & train = data->Y;
  double mean_value = data->mean_value;

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
  double b0 = 0.5 * noise->var_total;
  double aN = a0 + train.nonZeros() / 2.0;
  double bN = b0 + sumsq / 2.0;
  noise->alpha = rgamma(aN, 1.0 / bN);
  if (noise->alpha > noise->alpha_max) {
    noise->alpha = noise->alpha_max;
  }
}

template class INoiseModelDisp<FixedGaussianNoise>;
template class INoiseModelDisp<AdaptiveGaussianNoise>;
template class INoiseModelDisp<ProbitNoise>;
