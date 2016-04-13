#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "latentpriorvb.h"

using namespace std; 
using namespace Eigen;

void BPMFPriorVB::init(const int num_latent) {
  mu_mean.resize(num_latent);
  mu_var.resize(num_latent);
  mu_mean.setZero();
  mu_var.setOnes();

  lambda_a.resize(num_latent);
  lambda_b.resize(num_latent);
  lambda_a.setConstant(0.1);
  lambda_b.setConstant(0.1);

  // parameters of Normal-Gamma distribution
  lambda_a0 = 1.0;
  lambda_b0 = 1.0;
  b0 = 2.0;
}

void BPMFPriorVB::update_latents(
    Eigen::MatrixXd &Umean,
    Eigen::MatrixXd &Uvar,
    const Eigen::SparseMatrix<double> &Y,
    const double mean_value,
    const Eigen::MatrixXd &Vmean,
    const Eigen::MatrixXd &Vvar,
    const double alpha) {
}

void BPMFPriorVB::update_prior(const Eigen::MatrixXd &Umean, const Eigen::MatrixXd &Uvar) {
}
