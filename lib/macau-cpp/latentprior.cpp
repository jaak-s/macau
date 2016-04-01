#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cblas.h>
#include <math.h>
#include <omp.h>
#include <iomanip>

#include "mvnormal.h"
#include "macau.h"
#include "chol.h"
#include "linop.h"
extern "C" {
  #include <sparse.h>
}

using namespace std; 
using namespace Eigen;

/** BPMFPrior */
void BPMFPrior::sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                    const Eigen::MatrixXd &samples, double alpha, const int num_latent) {
  const int N = U.cols();
  
#pragma omp parallel for schedule(dynamic, 1)
  for(int n = 0; n < N; n++) {
    sample_latent_blas(U, n, mat, mean_value, samples, alpha, mu, Lambda, num_latent);
  }
}

void BPMFPrior::update_prior(const Eigen::MatrixXd &U) {
  tie(mu, Lambda) = CondNormalWishart(U, mu0, b0, WI, df);
}


void BPMFPrior::init(const int num_latent) {
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
}

/** MacauPrior */
template<class FType>
void MacauPrior<FType>::init(const int num_latent, FType * Fmat, bool comp_FtF) {
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
  F = std::unique_ptr<FType>(Fmat);
  use_FtF = comp_FtF;
  if (use_FtF) {
    FtF.resize(F->cols(), F->cols());
    At_mul_A(FtF, *F);
  }

  Uhat.resize(num_latent, F->rows());
  Uhat.setZero();

  beta.resize(num_latent, F->cols());
  beta.setZero();

  // initial value (should be determined automatically)
  lambda_beta = 5.0;
  // Hyper-prior for lambda_beta (mean 1.0, var of 1e+3):
  lambda_beta_mu0 = 1.0;
  lambda_beta_nu0 = 1e-3;
}

template<class FType>
void MacauPrior<FType>::setLambdaBeta(double lb) {
  lambda_beta = lb;
}

template<class FType>
void MacauPrior<FType>::sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                    const Eigen::MatrixXd &samples, double alpha, const int num_latent) {
  const int N = U.cols();
#pragma omp parallel for schedule(dynamic, 1)
  for(int n = 0; n < N; n++) {
    // TODO: try moving mu + Uhat.col(n) inside sample_latent for speed
    sample_latent_blas(U, n, mat, mean_value, samples, alpha, mu + Uhat.col(n), Lambda, num_latent);
  }
}

template<class FType>
void MacauPrior<FType>::update_prior(const Eigen::MatrixXd &U) {
  // residual (Uhat is later overwritten):
  Uhat.noalias() = U - Uhat;
  MatrixXd BBt = A_mul_At_combo(beta);
  // sampling Gaussian
  tie(mu, Lambda) = CondNormalWishart(Uhat, mu0, b0, WI + lambda_beta * BBt, df + beta.cols());
  sample_beta(U);
  compute_uhat(Uhat, *F, beta);
  lambda_beta = sample_lambda_beta(beta, Lambda, lambda_beta_nu0, lambda_beta_mu0);
}

template<class FType>
double MacauPrior<FType>::getLinkNorm() {
  return beta.norm();
}

/** Update beta and Uhat */
template<class FType>
void MacauPrior<FType>::sample_beta(const Eigen::MatrixXd &U) {
  const int num_feat = beta.cols();
  // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + sqrt(lambda_beta) * Normal(0, Lambda^-1)
  // Ft_y is [ D x F ] matrix
  MatrixXd tmp = (U + MvNormal_prec_omp(Lambda, U.cols())).colwise() - mu;
  MatrixXd Ft_y = A_mul_B(tmp, *F) + sqrt(lambda_beta) * MvNormal_prec_omp(Lambda, num_feat);

  if (use_FtF) {
    MatrixXd K(FtF.rows(), FtF.cols());
    K.triangularView<Eigen::Lower>() = FtF;
    for (int i = 0; i < K.cols(); i++) {
      K(i,i) += lambda_beta;
    }
    chol_decomp(K);
    chol_solve_t(K, Ft_y);
    beta = Ft_y;
  } else {
    // BlockCG
    solve_blockcg(beta, *F, lambda_beta, Ft_y, tol, 32, 8);
  }
}

std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu) {
  const int D = beta.rows();
  MatrixXd BB(D, D);
  A_mul_At_combo(BB, beta);
  double nux = nu + beta.rows() * beta.cols();
  double mux = mu * nux / (nu + mu * (BB.selfadjointView<Eigen::Lower>() * Lambda_u).trace() );
  double b   = nux / 2;
  double c   = 2 * mux / nux;
  return std::make_pair(b, c);
}

double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu) {
  auto gamma_post = posterior_lambda_beta(beta, Lambda_u, nu, mu);
  return rgamma(gamma_post.first, gamma_post.second);
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

void sample_latent_blas(MatrixXd &s, int mm, const SparseMatrix<double> &mat, double mean_rating,
    const MatrixXd &samples, double alpha, const VectorXd &mu_u, const MatrixXd &Lambda_u,
    const int num_latent)
{
  MatrixXd MM = Lambda_u;
  VectorXd rr = VectorXd::Zero(num_latent);
  for (SparseMatrix<double>::InnerIterator it(mat, mm); it; ++it) {
    auto col = samples.col(it.row());
    MM.triangularView<Eigen::Lower>() += alpha * col * col.transpose();
    rr.noalias() += col * ((it.value() - mean_rating) * alpha);
  }

  Eigen::LLT<MatrixXd> chol = MM.llt();
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

/**
 * X = A * B
 */
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  MatrixXd out(A.rows(), B.cols());
  A_mul_B_blas(out, A, B);
  return out;
}

Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseFeat & B) {
  MatrixXd out(A.rows(), B.cols());
  A_mul_Bt(out, B.Mt, A);
  return out;
}

Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseDoubleFeat & B) {
  MatrixXd out(A.rows(), B.cols());
  A_mul_Bt(out, B.Mt, A);
  return out;
}

template class MacauPrior<SparseFeat>;
template class MacauPrior<SparseDoubleFeat>;
//template class MacauPrior<Eigen::MatrixXd>;

