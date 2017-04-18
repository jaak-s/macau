#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <cmath>
#include <iostream>

#include "sparsetensor.h"
#include "bpmfutils.h"
#include "noisemodels.h"

using namespace Eigen;

////  AdaptiveGaussianNoise  ////
void AdaptiveGaussianNoise::init(MatrixData & matrixData) {
  double se = 0.0;
  double mean_value = matrixData.mean_value;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
  for (int k = 0; k < matrixData.Y.outerSize(); ++k) {
    for (SparseMatrix<double>::InnerIterator it(matrixData.Y, k); it; ++it) {
      se += square(it.value() - mean_value);
    }
  }

  var_total = se / matrixData.Y.nonZeros();
  if (var_total <= 0.0 || std::isnan(var_total)) {
    // if var cannot be computed using 1.0
    var_total = 1.0;
  }
  // Var(noise) = Var(total) / (SN + 1)
  alpha     = (sn_init + 1.0) / var_total;
  alpha_max = (sn_max + 1.0)  / var_total;
}

void AdaptiveGaussianNoise::init(TensorData & data) {
  double se = 0.0;
  double mean_value = data.mean_value;

  auto& sparseMode   = (*data.Y)[0];
  VectorXd & values  = sparseMode->values;

#pragma omp parallel for schedule(dynamic, 4) reduction(+:se)
  for (int i = 0; i < values.size(); i++) {
    se += square(values(i) - mean_value);
  }
  var_total = se / values.size();
  if (var_total <= 0.0 || std::isnan(var_total)) {
    var_total = 1.0;
  }
  // Var(noise) = Var(total) / (SN + 1)
  alpha     = (sn_init + 1.0) / var_total;
  alpha_max = (sn_max + 1.0)  / var_total;
}

void AdaptiveGaussianNoise::update(MatrixData & data, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples)
{
  double sumsq = 0.0;
  MatrixXd & U = *samples[0];
  MatrixXd & V = *samples[1];

  Eigen::SparseMatrix<double> & train = data.Y;
  double mean_value = data.mean_value;

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

void AdaptiveGaussianNoise::update(TensorData & data, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples)
{
  double sumsq = 0.0;
  double mean_value = data.mean_value;

  auto& sparseMode = (*data.Y)[0];
  auto& U = samples[0];

  const int nmodes = samples.size();
  const int num_latents = U->rows();

#pragma omp parallel for schedule(dynamic, 4) reduction(+:sumsq)
  for (int n = 0; n < data.dims(0); n++) {
    Eigen::VectorXd u = U->col(n);
    for (int j = sparseMode->row_ptr(n);
             j < sparseMode->row_ptr(n + 1);
             j++)
    {
      VectorXi idx = sparseMode->indices.row(j);
      // computing prediction from tensor
      double Yhat = mean_value;
      for (int d = 0; d < num_latents; d++) {
        double tmp = u(d);

        for (int m = 1; m < nmodes; m++) {
          tmp *= (*samples[m])(d, idx(m - 1));
        }
        Yhat += tmp;
      }
      sumsq += square(Yhat - sparseMode->values(j));
    }

  }
  double a0 = 0.5;
  double b0 = 0.5;
  double aN = a0 + sparseMode->values.size() / 2.0;
  double bN = b0 + sumsq / 2.0;
  alpha = rgamma(aN, 1.0 / bN);
  if (alpha > alpha_max) {
    alpha = alpha_max;
  }
}

inline double nCDF(double val) {return 0.5 * erfc(-val * M_SQRT1_2);}

/////  evalModel functions
void ProbitNoise::evalModel(MatrixData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var,
        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) {
  const unsigned N = data.Ytest.nonZeros();
  Eigen::VectorXd pred(N);
  Eigen::VectorXd test(N);
  Eigen::MatrixXd & rows = *samples[0];
  Eigen::MatrixXd & cols = *samples[1];

// #pragma omp parallel for schedule(dynamic,8) reduction(+:se, se_avg) <- dark magic :)
  for (int k = 0; k < data.Ytest.outerSize(); ++k) {
    int idx = data.Ytest.outerIndexPtr()[k];
    for (Eigen::SparseMatrix<double>::InnerIterator it(data.Ytest,k); it; ++it) {
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

void FixedGaussianNoise::evalModel(MatrixData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var,
        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) {
   auto rmse = eval_rmse(data.Ytest, n, predictions, predictions_var, *samples[1], *samples[0], data.mean_value);
   rmse_test = rmse.second;
   rmse_test_onesample = rmse.first;
}

void AdaptiveGaussianNoise::evalModel(MatrixData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var,
        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) {
   auto rmse = eval_rmse(data.Ytest, n, predictions, predictions_var, *samples[1], *samples[0], data.mean_value);
   rmse_test = rmse.second;
   rmse_test_onesample = rmse.first;
}


// evalModel for TensorData
void ProbitNoise::evalModel(TensorData & data, const int Nepoch, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var,
        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) {
  // TODO
  throw std::runtime_error("ProbitNoise::evalModel unimplemented.");
}


void FixedGaussianNoise::evalModel(TensorData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var,
        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) {
  auto rmse = eval_rmse_tensor(data.Ytest, n, predictions, predictions_var, samples, data.mean_value);
  rmse_test = rmse.second;
  rmse_test_onesample = rmse.first;
}

void AdaptiveGaussianNoise::evalModel(TensorData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var,
        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) {
  auto rmse = eval_rmse_tensor(data.Ytest, n, predictions, predictions_var, samples, data.mean_value);
  rmse_test = rmse.second;
  rmse_test_onesample = rmse.first;
}
