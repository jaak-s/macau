#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "macau.h"

using namespace std; 
using namespace Eigen;

void BPMFPrior::sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                    const Eigen::MatrixXd &samples, double alpha, const int num_latent) {
  const int N = U.cols();
#pragma omp parallel for
  for(int n = 0; n < N; n++) {
    sample_latent(U, n, mat, mean_value, samples, alpha, mu, Lambda, num_latent);
  }
}

void BPMFPrior::update_prior(const Eigen::MatrixXd &U) {
  tie(mu, Lambda) = CondNormalWishart(U, mu0, b0, WI, df);
}
