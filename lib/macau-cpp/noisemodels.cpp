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

 // Evaluation metrics
void FixedGaussianNoise::evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, const Eigen::MatrixXd &cols, const Eigen::MatrixXd &rows, double mean_rating) {
   auto rmse = eval_rmse(Ytest, n, predictions, predictions_var, cols, rows, mean_rating);
   rmse_test = rmse.second;
   rmse_test_onesample = rmse.first;
}

void AdaptiveGaussianNoise::evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, const Eigen::MatrixXd &cols, const Eigen::MatrixXd &rows, double mean_rating) {
   auto rmse = eval_rmse(Ytest, n, predictions, predictions_var, cols, rows, mean_rating);
   rmse_test = rmse.second;
   rmse_test_onesample = rmse.first;
}

inline double nCDF(double val) {return 0.5 * erfc(-val * M_SQRT1_2);}

void ProbitNoise::evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, const Eigen::MatrixXd &cols, const Eigen::MatrixXd &rows, double mean_rating) {
  const unsigned N = Ytest.nonZeros();
  Eigen::VectorXd pred(N);
  Eigen::VectorXd test(N);

// #pragma omp parallel for schedule(dynamic,8) reduction(+:se, se_avg) <- dark magic :)
  for (int k = 0; k < Ytest.outerSize(); ++k) {
    int idx = Ytest.outerIndexPtr()[k];
    for (Eigen::SparseMatrix<double>::InnerIterator it(Ytest,k); it; ++it) {
     pred[idx] = nCDF(cols.col(it.col()).dot(rows.col(it.row())));
     test[idx] = it.value();

      // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
      double pred_avg;
      if (n == 0) {
        pred_avg = pred[idx];
      } else {
        double delta = pred[idx] - predictions[idx];
        pred_avg = (predictions[idx] + delta / (n + 1));
        predictions_var[idx] += delta * (pred[idx] - pred_avg);
      }
      predictions[idx++] = pred_avg;

   }
  }
  auc_test_onesample = auc(pred,test);
  auc_test = auc(predictions, test);
}

template class INoiseModelDisp<FixedGaussianNoise>;
template class INoiseModelDisp<AdaptiveGaussianNoise>;
template class INoiseModelDisp<ProbitNoise>;
