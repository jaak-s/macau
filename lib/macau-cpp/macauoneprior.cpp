#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <math.h>
#include <omp.h>
#include <iostream>

#include "mvnormal.h"
#include "linop.h"
#include "bpmfutils.h"
#include "macauoneprior.h"

using namespace std; 
using namespace Eigen;

template<class FType>
void MacauOnePrior<FType>::init(const int nlatent, std::unique_ptr<FType> &Fmat) {
  num_latent = nlatent;

  // parameters of Normal-Gamma distributions
  mu     = VectorXd::Constant(num_latent, 0.0);
  lambda = VectorXd::Constant(num_latent, 10.0);
  // their hyperparameter (lambda_0)
  l0 = 2.0;
  lambda_a0 = 1.0;
  lambda_b0 = 1.0;

  // side information
  F       = std::move(Fmat);
  F_colsq = col_square_sum(*F);

  Uhat = MatrixXd::Constant(num_latent, F->rows(), 0.0);
  beta = MatrixXd::Constant(num_latent, F->cols(), 0.0);

  // initial value (should be determined automatically)
  // Hyper-prior for lambda_beta (mean 1.0):
  lambda_beta     = VectorXd::Constant(num_latent, 5.0);
  lambda_beta_mu0 = 1.0;
  lambda_beta_nu0 = 1e-3;
}

template<class FType>
void MacauOnePrior<FType>::sample_latents(
    Eigen::MatrixXd &U,
    const Eigen::SparseMatrix<double> &Ymat,
    double mean_value,
    const Eigen::MatrixXd &V,
    double alpha,
    const int num_latent)
{
  const int N = U.cols();
  const int D = U.rows();

#pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < N; i++) {

    const int nnz = Ymat.outerIndexPtr()[i + 1] - Ymat.outerIndexPtr()[i];
    VectorXd Yhat(nnz);

    // precalculating Yhat and Qi
    int idx = 0;
    VectorXd Qi = lambda;
    for (SparseMatrix<double>::InnerIterator it(Ymat, i); it; ++it, idx++) {
      Qi.noalias() += alpha * V.col(it.row()).cwiseAbs2();
      Yhat(idx)     = mean_value + U.col(i).dot( V.col(it.row()) );
    }
    VectorXd rnorms(num_latent);
    bmrandn_single(rnorms);

    for (int d = 0; d < D; d++) {
      // computing Lid
      const double uid = U(d, i);
      double Lid = lambda(d) * (mu(d) + Uhat(d, i));

      idx = 0;
      for ( SparseMatrix<double>::InnerIterator it(Ymat, i); it; ++it, idx++) {
        const double vjd = V(d, it.row());
        // L_id += alpha * (Y_ij - k_ijd) * v_jd
        Lid += alpha * (it.value() - (Yhat(idx) - uid*vjd)) * vjd;
      }
      // Now use Lid and Qid to update uid
      double uid_old = U(d, i);
      double uid_var = 1.0 / Qi(d);

      // sampling new u_id ~ Norm(Lid / Qid, 1/Qid)
      U(d, i) = Lid * uid_var + sqrt(uid_var) * rnorms(d);

      // updating Yhat
      double uid_delta = U(d, i) - uid_old;
      idx = 0;
      for (SparseMatrix<double>::InnerIterator it(Ymat, i); it; ++it, idx++) {
        Yhat(idx) += uid_delta * V(d, it.row());
      }
    }
  }
}

template<class FType>
void MacauOnePrior<FType>::update_prior(const Eigen::MatrixXd &U) {
  sample_mu_lambda(U);
  /*
  sample_beta(U);
  compute_uhat(Uhat, *F, beta);
  sample_lambda_beta();
  */
}

template<class FType>
void MacauOnePrior<FType>::sample_mu_lambda(const Eigen::MatrixXd &U) {
  // TODO: use U - Uhat
  MatrixXd Lambda(num_latent, num_latent);
  MatrixXd WI(num_latent, num_latent);
  WI.setIdentity();

  tie(mu, Lambda) = CondNormalWishart(U, VectorXd::Constant(num_latent, 0.0), 2.0, WI, num_latent);
  lambda = Lambda.diagonal();
}

template<class FType>
void MacauOnePrior<FType>::sample_beta(const Eigen::MatrixXd &U) {
}

template<class FType>
void MacauOnePrior<FType>::sample_lambda_beta() {
  // TODO
}

template class MacauOnePrior<SparseFeat>;
template class MacauOnePrior<SparseDoubleFeat>;
