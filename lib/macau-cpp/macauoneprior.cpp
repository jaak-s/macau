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
  lambda_beta_a0 = 0.1;
  lambda_beta_b0 = 0.1;
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
void MacauOnePrior<FType>::sample_latents(
        ProbitNoise& noiseModel,
        TensorData & data,
        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
        int mode,
        const int num_latent)
{
  throw std::runtime_error("Unimplemented: sample_latents");
}

template<class FType>
void MacauOnePrior<FType>::sample_latents(
        double noisePrecision,
        TensorData & data,
        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
        int mode,
        const int num_latent)
{
  auto& sparseMode = (*data.Y)[mode];
  auto& U = samples[mode];
  const int N = U->cols();
  const int D = num_latent;
  VectorView<Eigen::MatrixXd> view(samples, mode);
  const int nmodes1 = view.size();
  const double mean_value = data.mean_value;

  if (U->rows() != num_latent) {
    throw std::runtime_error("U->rows() must be equal to num_latent.");
  }

  Eigen::VectorXi & row_ptr = sparseMode->row_ptr;
  Eigen::MatrixXi & indices = sparseMode->indices;
  Eigen::VectorXd & values  = sparseMode->values;

#pragma omp parallel for schedule(dynamic, 8)
  for (int i = 0; i < N; i++) {
    // looping over all non-zeros for row i of the mode
    // precalculating Yhat and Qi
    const int nnz = row_ptr(i + 1) - row_ptr(i);
    VectorXd Yhat(nnz);
    VectorXd tmpd(nnz);
    VectorXd Qi = lambda;

    for (int idx = 0; idx < nnz; idx++) {
      int j = idx + row_ptr(i);
      VectorXd prod = VectorXd::Ones(D);
      for (int m = 0; m < nmodes1; m++) {
        auto v = view.get(m)->col(indices(j, m));
        prod.noalias() = prod.cwiseProduct(v);
      }
      Qi.noalias() += noisePrecision * prod.cwiseAbs2();
      Yhat(idx) = mean_value + U->col(i).dot(prod);
    }

    // generating random numbers
    VectorXd rnorms(num_latent);
    bmrandn_single(rnorms);

    for (int d = 0; d < D; d++) {
      // computing Lid
      const double uid = (*U)(d, i);
      double Lid = lambda(d) * (mu(d) + Uhat(d, i));
      
      for (int idx = 0; idx < nnz; idx++) {
        int j = idx + row_ptr(i);

        // computing t = vjd * wkd * ..
        double t = 1.0;
        for (int m = 0; m < nmodes1; m++) {
          t *= (*view.get(m))(d, indices(j, m));
        }
        tmpd(idx) = t;
        // L_id += alpha * (Y_ijk - k_ijkd) * v_jd * wkd
        Lid += noisePrecision * (values(j) - (Yhat(idx) - uid * t)) * t;
      }
      // Now use Lid and Qid to update uid
      double uid_old = uid;
      double uid_var = 1.0 / Qi(d);

      // sampling new u_id ~ Norm(Lid / Qid, 1/Qid)
      (*U)(d, i) = Lid * uid_var + sqrt(uid_var) * rnorms(d);

      // updating Yhat
      double uid_delta = (*U)(d, i) - uid_old;
      for (int idx = 0; idx < nnz; idx++) {
        Yhat(idx) += uid_delta * tmpd(idx);
      }
    }
  }
}

template<class FType>
void MacauOnePrior<FType>::update_prior(const Eigen::MatrixXd &U) {
  sample_mu_lambda(U);
  sample_beta(U);
  compute_uhat(Uhat, *F, beta);
  sample_lambda_beta();
}

template<class FType>
void MacauOnePrior<FType>::sample_mu_lambda(const Eigen::MatrixXd &U) {
  MatrixXd Lambda(num_latent, num_latent);
  MatrixXd WI(num_latent, num_latent);
  WI.setIdentity();
  int N = U.cols();

  MatrixXd Udelta(num_latent, N);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++) {
    for (int d = 0; d < num_latent; d++) {
      Udelta(d, i) = U(d, i) - Uhat(d, i);
    }
  }
  tie(mu, Lambda) = CondNormalWishart(Udelta, VectorXd::Constant(num_latent, 0.0), 2.0, WI, num_latent);
  lambda = Lambda.diagonal();
}

template<class FType>
void MacauOnePrior<FType>::sample_beta(const Eigen::MatrixXd &U) {
  // updating beta and beta_var
  const int nfeat = beta.cols();
  const int N = U.cols();
  const int blocksize = 4;

  MatrixXd Z;

#pragma omp parallel for private(Z) schedule(static, 1)
  for (int dstart = 0; dstart < num_latent; dstart += blocksize) {
    const int dcount = std::min(blocksize, num_latent - dstart);
    Z.resize(dcount, U.cols());

    for (int i = 0; i < N; i++) {
      for (int d = 0; d < dcount; d++) {
        int dx = d + dstart;
        Z(d, i) = U(dx, i) - mu(dx) - Uhat(dx, i);
      }
    }

    for (int f = 0; f < nfeat; f++) {
      VectorXd zx(dcount), delta_beta(dcount), randvals(dcount);
      // zx = Z[dstart : dstart + dcount, :] * F[:, f]
      At_mul_Bt(zx, *F, f, Z);
      // TODO: check if sampling randvals for whole [nfeat x dcount] matrix works faster
      bmrandn_single( randvals );

      for (int d = 0; d < dcount; d++) {
        int dx = d + dstart;
        double A_df     = lambda_beta(dx) + lambda(dx) * F_colsq(f);
        double B_df     = lambda(dx) * (zx(d) + beta(dx,f) * F_colsq(f));
        double A_inv    = 1.0 / A_df;
        double beta_new = B_df * A_inv + sqrt(A_inv) * randvals(d);
        delta_beta(d)   = beta(dx,f) - beta_new;

        beta(dx, f)     = beta_new;
      }
      // Z[dstart : dstart + dcount, :] += F[:, f] * delta_beta'
      add_Acol_mul_bt(Z, *F, f, delta_beta);
    }
  }
}

template<class FType>
void MacauOnePrior<FType>::sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                          double mean_value, const Eigen::MatrixXd &samples, const int num_latent) {
 //TODO
 throw std::runtime_error("Not implemented!");
}

template<class FType>
void MacauOnePrior<FType>::sample_lambda_beta() {
  double lambda_beta_a = lambda_beta_a0 + beta.cols() / 2.0;
  VectorXd lambda_beta_b = VectorXd::Constant(beta.rows(), lambda_beta_b0);
  const int D = beta.rows();
  const int F = beta.cols();
#pragma omp parallel
  {
    VectorXd tmp(D);
    tmp.setZero();
#pragma omp for schedule(static)
    for (int f = 0; f < F; f++) {
      for (int d = 0; d < D; d++) {
        tmp(d) += square(beta(d, f));
      }
    }
#pragma omp critical
    {
      lambda_beta_b += tmp / 2;
    }
  }
  for (int d = 0; d < D; d++) {
    lambda_beta(d) = rgamma(lambda_beta_a, 1.0 / lambda_beta_b(d));
  }
}

template<class FType>
void MacauOnePrior<FType>::saveModel(std::string prefix) {
  writeToCSVfile(prefix + "-latentmean.csv", mu);
  writeToCSVfile(prefix + "-link.csv", beta);
}

template class MacauOnePrior<SparseFeat>;
template class MacauOnePrior<SparseDoubleFeat>;
