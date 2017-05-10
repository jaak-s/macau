#pragma once

#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <memory>

#include "sparsetensor.h"

inline double tick() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count(); 
}

inline double clamp(double x, double min, double max) {
  return x < min ? min : (x > max ? max : x);
}

inline std::pair<double, double> getMinMax(const Eigen::SparseMatrix<double> &mat) { 
    double min = INFINITY;
    double max = -INFINITY;
    for (int k = 0; k < mat.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(mat,k); it; ++it) {
            double v = it.value();
            if (v < min) min = v;
            if (v > max) max = v;
        }
    }
    return std::make_pair(min, max);
}

inline void split_work_mpi(int num_latent, int num_nodes, int* work) {
   double avg_work = num_latent / (double) num_nodes;
   int work_unit;
   if (2 <= avg_work) work_unit = 2;
   else work_unit = 1;

   int min_work  = work_unit * (int)floor(avg_work / work_unit);
   int work_left = num_latent;

   for (int i = 0; i < num_nodes; i++) {
      work[i]    = min_work;
      work_left -= min_work;
   }
   int i = 0;
   while (work_left > 0) {
      int take = std::min(work_left, work_unit);
      work[i]   += take;
      work_left -= take;
      i = (i + 1) % num_nodes;
   }
}

inline void sparseFromIJV(Eigen::SparseMatrix<double> &X, int* rows, int* cols, double* values, int N) {
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(N);
  for (int n = 0; n < N; n++) {
    tripletList.push_back(T(rows[n], cols[n], values[n]));
  }
  X.setFromTriplets(tripletList.begin(), tripletList.end());
}

inline void sparseFromIJV(Eigen::SparseMatrix<double> &X, Eigen::MatrixXi &idx, Eigen::VectorXd &values) {
  if (idx.rows() != values.size()) {
    throw std::runtime_error("sparseFromIJV: idx.rows() must equal values.size().");
  }
  if (idx.cols() != 2) {
    throw std::runtime_error("sparseFromIJV: idx.cols() must be equal to 2.");
  }
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  int N = values.size();
  tripletList.reserve(N);
  for (int n = 0; n < N; n++) {
    tripletList.push_back(T(idx(n, 0), idx(n, 1), values(n)));
  }
  X.setFromTriplets(tripletList.begin(), tripletList.end());
}

inline double square(double x) { return x * x; }

inline std::pair<double,double> eval_rmse(Eigen::SparseMatrix<double> & P, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, const Eigen::MatrixXd &sample_m, const Eigen::MatrixXd &sample_u, double mean_rating)
{
  double se = 0.0, se_avg = 0.0;
#pragma omp parallel for schedule(dynamic,8) reduction(+:se, se_avg)
  for (int k = 0; k < P.outerSize(); ++k) {
    int idx = P.outerIndexPtr()[k];
    for (Eigen::SparseMatrix<double>::InnerIterator it(P,k); it; ++it) {
      const double pred = sample_m.col(it.col()).dot(sample_u.col(it.row())) + mean_rating;
      se += square(it.value() - pred);

      // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
      double pred_avg;
      if (n == 0) {
        pred_avg = pred;
      } else {
        double delta = pred - predictions[idx];
        pred_avg = (predictions[idx] + delta / (n + 1));
        predictions_var[idx] += delta * (pred - pred_avg);
      }
      se_avg += square(it.value() - pred_avg);
      predictions[idx++] = pred_avg;
    }
  }

  const unsigned N = P.nonZeros();
  const double rmse = sqrt( se / N );
  const double rmse_avg = sqrt( se_avg / N );
  return std::make_pair(rmse, rmse_avg);
}

std::pair<double,double> eval_rmse_tensor(
		SparseMode & sparseMode,
		const int Nepoch,
		Eigen::VectorXd & predictions,
		Eigen::VectorXd & predictions_var,
		std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
		double mean_value);

inline void row_mean_var(Eigen::VectorXd & mean, Eigen::VectorXd & var, const Eigen::MatrixXd X) {
  const int N = X.cols();
  const int D = X.rows();

  mean.resize(D);
  var.resize(D);
  mean.setZero();
  var.setZero();

#pragma omp parallel
  {
    Eigen::VectorXd tmp(D);
    tmp.setZero();
#pragma omp for schedule(static)
    for (int i = 0; i < N; i++) {
      for (int d = 0; d < D; d++) {
        tmp(d) += X(d, i);
      }
    }
#pragma omp critical
    {
      mean += tmp;
    }
  }
  // computing mean
  mean /= N;

#pragma omp parallel
  {
    Eigen::VectorXd tmp(D);
    tmp.setZero();
#pragma omp for schedule(static)
    for (int i = 0; i < N; i++) {
      for (int d = 0; d < D; d++) {
        tmp(d) += square(X(d, i) - mean(d));
      }
    }
#pragma omp critical
    {
      var += tmp;
    }
  }
  var /= N;
}

inline void writeToCSVfile(std::string filename, Eigen::MatrixXd matrix) {
  const static Eigen::IOFormat csvFormat(6, Eigen::DontAlignCols, ",", "\n");
  std::ofstream file(filename.c_str());
  file << matrix.format(csvFormat);
}

inline std::string to_string_with_precision(const double a_value, const int n = 6)
{
    std::ostringstream out;
    out << std::setprecision(n) << a_value;
    return out.str();
}

inline double auc(Eigen::VectorXd & pred, Eigen::VectorXd & test)
{
	Eigen::VectorXd stack_x(pred.size());
	Eigen::VectorXd stack_y(pred.size());
	double auc = 0.0;

	if (pred.size() == 0) {
		return NAN;
	}

	std::vector<unsigned int> permutation( pred.size() );
	for(unsigned int i = 0; i < pred.size(); i++) {
		permutation[i] = i;
	}
	std::sort(permutation.begin(), permutation.end(), [&pred](unsigned int a, unsigned int b) { return pred[a] < pred[b];});

	double NP = test.sum();
	double NN = test.size() - NP;
	//Build stack_x and stack_y
	stack_x[0] = test[permutation[0]];
	stack_y[0] = 1-stack_x[0];
	for(int i=1; i < pred.size(); i++) {
		stack_x[i] = stack_x[i-1] + test[permutation[i]];
		stack_y[i] = stack_y[i-1] + 1 - test[permutation[i]];
	}

	for(int i=0; i < pred.size() - 1; i++) {
		auc += (stack_x(i+1) - stack_x(i)) * stack_y(i+1); //TODO:Make it Eigen
	}

	return auc / (NP*NN);
}
