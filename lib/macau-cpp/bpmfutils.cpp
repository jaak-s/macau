#include <Eigen/Dense>
#include <memory>
#include <iostream>
#include <limits>
#include "bpmfutils.h"
#include "sparsetensor.h"

using namespace Eigen;

std::pair<double,double> eval_rmse_tensor(
		SparseMode & sparseMode,
		const int Nepoch,
		Eigen::VectorXd & predictions,
		Eigen::VectorXd & predictions_var,
		std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
		double mean_value)
{
  auto& U = samples[0];

  const int nmodes = samples.size();
  const int num_latents = U->rows();

  const unsigned N = sparseMode.values.size();
  double se = 0.0, se_avg = 0.0;

  if (N == 0) {
    // No test data, returning NaN's
    return std::make_pair(std::numeric_limits<double>::quiet_NaN(),
                          std::numeric_limits<double>::quiet_NaN());
  }

  if (N != predictions.size()) {
    throw std::runtime_error("Ytest.size() and predictions.size() must be equal.");
  }
	if (sparseMode.row_ptr.size() - 1 != U->cols()) {
    throw std::runtime_error("U.cols() and sparseMode size must be equal.");
	}

#pragma omp parallel for schedule(dynamic, 2) reduction(+:se, se_avg)
  for (int n = 0; n < U->cols(); n++) {
    Eigen::VectorXd u = U->col(n);
    for (int j = sparseMode.row_ptr(n);
             j < sparseMode.row_ptr(n + 1);
             j++)
    {
      VectorXi idx = sparseMode.indices.row(j);
      double pred = mean_value;
      for (int d = 0; d < num_latents; d++) {
        double tmp = u(d);

        for (int m = 1; m < nmodes; m++) {
          tmp *= (*samples[m])(d, idx(m - 1));
        }
        pred += tmp;
      }

      double pred_avg;
      if (Nepoch == 0) {
        pred_avg = pred;
      } else {
        double delta = pred - predictions(j);
        pred_avg = (predictions(j) + delta / (Nepoch + 1));
        predictions_var(j) += delta * (pred - pred_avg);
      }
      se     += square(sparseMode.values(j) - pred);
      se_avg += square(sparseMode.values(j) - pred_avg);
      predictions(j) = pred_avg;
    }
  }
  const double rmse = sqrt(se / N);
  const double rmse_avg = sqrt(se_avg / N);
  return std::make_pair(rmse, rmse_avg);
}
