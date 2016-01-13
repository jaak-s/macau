#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <math.h>

#include "mvnormal.h"
#include "macau.h"

using namespace std; 
using namespace Eigen;

/** BPMFPrior */
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

/** MacauPrior */
void MacauPrior::init(const int num_latent, Eigen::MatrixXd & Fmat, bool comp_FtF) {
  mu.resize(num_latent);
  mu.setZero();

  Lambda.resize(num_latent, num_latent);
  Lambda.setIdentity();
  Lambda *= 10;

  // parameters of Inv-Whishart distribution
  WI.resize(num_latent, num_latent);
  WI.setIdentity();
  mu0.resize(num_latent);
  mu0.setZero();
  b0 = 2;
  df = num_latent;

  // side information
  F = Fmat;
  use_FtF = comp_FtF;
  if (use_FtF) {
    At_mul_A(FtF, F);
  }

  Uhat.resize(num_latent, F.rows());
  Uhat.setZero();

  beta.resize(F.cols(), num_latent);
  beta.setZero();
}

void MacauPrior::sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                    const Eigen::MatrixXd &samples, double alpha, const int num_latent) {
  const int N = U.cols();
#pragma omp parallel for
  for(int n = 0; n < N; n++) {
    // TODO: try moving mu + Uhat.col(n) inside sample_latent for speed
    sample_latent(U, n, mat, mean_value, samples, alpha, mu + Uhat.col(n), Lambda, num_latent);
  }
}

void MacauPrior::update_prior(const Eigen::MatrixXd &U) {
  // residual:
  Uhat = U - Uhat;
  tie(mu, Lambda) = CondNormalWishart(Uhat, mu0, b0, WI + lambda_beta * (beta.transpose() * beta), df + beta.rows());
  // update beta and Uhat:
  sample_beta(U);
}

void MacauPrior::sample_beta(const Eigen::MatrixXd &U) {
  const int num_feat = beta.rows();
  // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + sqrt(lambda_beta) * Normal(0, Lambda^-1)
  MatrixXd Ft_y = A_mul_B(U + MvNormal_prec(Lambda, -mu, U.cols()), F) + sqrt(lambda_beta) * MvNormal_prec(Lambda, num_feat);

  if (use_FtF) {
  } else {
  }
}

/** global function */
void sample_latent(MatrixXd &s, int mm, const SparseMatrix<double> &mat, double mean_rating,
    const MatrixXd &samples, double alpha, const VectorXd &mu_u, const MatrixXd &Lambda_u,
    const int num_latent)
{
  // TODO: add cholesky update version
  MatrixXd MM = MatrixXd::Zero(num_latent, num_latent);
  VectorXd rr = VectorXd::Zero(num_latent);
  for (SparseMatrix<double>::InnerIterator it(mat, mm); it; ++it) {
    auto col = samples.col(it.row());
    MM.noalias() += col * col.transpose();
    rr.noalias() += col * ((it.value() - mean_rating) * alpha);
  }

  Eigen::LLT<MatrixXd> chol = (Lambda_u + alpha * MM).llt();
  if(chol.info() != Eigen::Success) {
    throw std::runtime_error("Cholesky Decomposition failed!");
  }

  rr.noalias() += Lambda_u * mu_u;
  chol.matrixL().solveInPlace(rr);
  for (int i = 0; i < num_latent; i++) {
    rr[i] += randn0();
  }
  chol.matrixU().solveInPlace(rr);
  s.col(mm).noalias() = rr;
}

void At_mul_A(Eigen::MatrixXd & result, const Eigen::MatrixXd & F) {
  result = F.transpose() * F;
}

Eigen::MatrixXd A_mul_B(const Eigen::MatrixXd & A, const Eigen::MatrixXd & B) {
  return A * B;
}
