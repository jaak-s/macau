#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>

#include "latentpriorvb.h"
#include "linop.h"

using namespace std; 
using namespace Eigen;

void BPMFPriorVB::init(const int num_latent, const double u_expected_squares) {
  mu_mean.resize(num_latent);
  mu_var.resize(num_latent);
  mu_mean.setZero();
  mu_var.setOnes();

  // parameters of Normal-Gamma distribution
  lambda_a0 = 1.0;
  lambda_b0 = 1.0;
  b0 = 2.0;

  lambda_b.resize(num_latent);
  lambda_b.setConstant(lambda_b0 + 0.5 * u_expected_squares);
}

Eigen::VectorXd BPMFPriorVB::getElambda(int N) {
  double lambda_a  = lambda_a0 + (N + 1.0) / 2.0;
  return VectorXd::Constant(lambda_b.size(), lambda_a).cwiseQuotient( lambda_b );
}

void BPMFPriorVB::update_latents(
    Eigen::MatrixXd &Umean,
    Eigen::MatrixXd &Uvar,
    const Eigen::SparseMatrix<double> &Ymat,
    const double mean_value,
    const Eigen::MatrixXd &Vmean,
    const Eigen::MatrixXd &Vvar,
    const double alpha)
{
  const int N = Umean.cols();
  const int D = Umean.rows();

  // compute E[lambda] .* E[mu]
  const VectorXd Elambda     = getElambda(N);
  const VectorXd Elambda_Emu = Elambda.cwiseProduct( mu_mean );

  // Vsq = E[ V^2 ] = E[V] .* E[V] + Var(V)
  MatrixXd Vsq( Vmean.rows(), Vmean.cols() );
#pragma omp parallel for schedule(static)
  for (int col = 0; col < Vmean.cols(); col++) {
    for (int row = 0; row < Vmean.rows(); row++) {
      Vsq(row, col) = Vmean(row, col) * Vmean(row, col) + Vvar(row, col);
    }
  }

#pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < N; i++) {
    VectorXd Qi = Elambda;

    const int nnz = Ymat.outerIndexPtr()[i + 1] - Ymat.outerIndexPtr()[i];
    VectorXd Yhat(nnz);

    // precalculating Yhat and Qi
    int idx = 0;
    for (SparseMatrix<double>::InnerIterator it(Ymat, i); it; ++it, idx++) {
      Qi += alpha * Vsq.col(it.row());
      Yhat(idx) = mean_value + Umean.col(i).dot( Vmean.col(it.row()) );
    }
    for (int d = 0; d < D; d++) {
      // computing Lid
      const double uid = Umean(d, i);
      double Lid = Elambda_Emu(d);

      idx = 0;
      for ( SparseMatrix<double>::InnerIterator it(Ymat, i); it; ++it, idx++) {
        const double vjd = Vmean(d, it.row());
        // Lid += alpha * (Yij - E[kijd]) * E[vjd]
        Lid += alpha * (it.value() - (Yhat(idx) - uid*vjd)) * vjd;
      }
      // Now use Lid and Qid to update uid
      double uid_old = Umean(d, i);
      double uid_var = 1.0 / Qi(d);
      Umean(d, i) = Lid * uid_var;
      Uvar(d, i)  = uid_var;
      // updating Yhat
      double uid_delta = Umean(d, i) - uid_old;
      idx = 0;
      for (SparseMatrix<double>::InnerIterator it(Ymat, i); it; ++it, idx++) {
        Yhat(idx) += uid_delta * Vmean(d, it.row());
      }
    }
  }
}

void BPMFPriorVB::update_prior(Eigen::MatrixXd &Umean, Eigen::MatrixXd &Uvar) {
  // TODO: parallelize or turn on Eigen's parallelism
  assert(Umean.rows() == Uvar.rows());
  assert(Vmean.rows() == Vvar.rows());
  const int N = Umean.cols();
  // updating mu_d
  VectorXd Elambda = getElambda(N);
  VectorXd A = Elambda * (b0 + N);
  VectorXd B = Elambda.cwiseProduct( Umean.rowwise().sum() );
  mu_mean = B.cwiseQuotient(A);
 
  for (int i = 0; i < A.size(); i++) {
    mu_var(i) = 1.0 / A(i);
  }

  // updating lambda_b
  lambda_b.setConstant(lambda_b0);
  // += 0.5 * b0 E[mu_d^2]
  lambda_b += 0.5 * b0 * (mu_mean.cwiseProduct(mu_mean) + mu_var);
  // += 0.5 * sum_i (E[uid] - E[mu_d])^2
  lambda_b += 0.5 * (Umean.colwise() - mu_mean).cwiseProduct(Umean.colwise() - mu_mean).rowwise().sum();
  // += 0.5 * sum_i (Var[uid])
  lambda_b += 0.5 * Uvar.rowwise().sum();
  lambda_b += 0.5 * mu_var * N;
}


// ------------- MacauPriorVB  ----------------


template<class FType>
void MacauPriorVB<FType>::init(const int num_latent, std::unique_ptr<FType> & Fmat, double usq) {
  mu_mean.resize(num_latent);
  mu_var.resize(num_latent);
  mu_mean.setZero();
  mu_var.setOnes();

  // parameters of Normal-Gamma distribution
  lambda_a0 = 1.0;
  lambda_b0 = 1.0;
  b0 = 2.0;

  lambda_b.resize(num_latent);
  lambda_b.setConstant(lambda_b0 + 0.5 * usq);

  // side information
  F = std::move(Fmat);
  F_colsq = col_square_sum(*F);

  Uhat.resize(num_latent, F->rows());
  Uhat.setZero();
  Uhat_valid = true;

  beta.resize(num_latent, F->cols());
  beta.setZero();

  beta_var.resize(num_latent, F->cols());
  beta_var.setOnes();

  // initial value (should be determined automatically)
  // Hyper-prior for lambda_beta (mean 1.0, var of 1e+3):
  lambda_beta_a  = VectorXd::Constant(num_latent, 0.5);
  lambda_beta_b  = VectorXd::Constant(num_latent, 0.1);
  lambda_beta_a0 = 0.001;     // Hyper-prior for lambda_beta
  lambda_beta_b0 = 0.001;     // Hyper-prior for lambda_beta
}

template<class FType>
void MacauPriorVB<FType>::update_latents(
    Eigen::MatrixXd &Umean,
    Eigen::MatrixXd &Uvar,
    const Eigen::SparseMatrix<double> &Ymat,
    const double mean_value,
    const Eigen::MatrixXd &Vmean,
    const Eigen::MatrixXd &Vvar,
    const double alpha)
{
  const int N = Umean.cols();
  const int D = Umean.rows();

  // compute E[lambda] .* E[mu]
  const VectorXd Elambda     = getElambda(N);
  const VectorXd Elambda_Emu = Elambda.cwiseProduct( mu_mean );

  update_uhat();

  // Vsq = E[ V^2 ] = E[V] .* E[V] + Var(V)
  MatrixXd Vsq( Vmean.rows(), Vmean.cols() );
#pragma omp parallel for schedule(static)
  for (int col = 0; col < Vmean.cols(); col++) {
    for (int row = 0; row < Vmean.rows(); row++) {
      Vsq(row, col) = Vmean(row, col) * Vmean(row, col) + Vvar(row, col);
    }
  }

#pragma omp parallel for schedule(dynamic, 4)
  for (int i = 0; i < N; i++) {
    VectorXd Qi = Elambda;

    const int nnz = Ymat.outerIndexPtr()[i + 1] - Ymat.outerIndexPtr()[i];
    VectorXd Yhat(nnz);

    // precalculating Yhat and Qi
    int idx = 0;
    for (SparseMatrix<double>::InnerIterator it(Ymat, i); it; ++it, idx++) {
      Qi += alpha * Vsq.col(it.row());
      Yhat(idx) = mean_value + Umean.col(i).dot( Vmean.col(it.row()) );
    }
    for (int d = 0; d < D; d++) {
      // computing Lid
      const double uid = Umean(d, i);
      double Lid = Elambda_Emu(d) + Uhat(d, i) * Elambda(d);

      idx = 0;
      for ( SparseMatrix<double>::InnerIterator it(Ymat, i); it; ++it, idx++) {
        const double vjd = Vmean(d, it.row());
        // Lid += alpha * (Yij - E[kijd]) * E[vjd]
        Lid += alpha * (it.value() - (Yhat(idx) - uid*vjd)) * vjd;
      }
      // Now use Lid and Qid to update uid
      double uid_old = Umean(d, i);
      double uid_var = 1.0 / Qi(d);
      Umean(d, i) = Lid * uid_var;
      Uvar(d, i)  = uid_var;
      // updating Yhat
      double uid_delta = Umean(d, i) - uid_old;
      idx = 0;
      for (SparseMatrix<double>::InnerIterator it(Ymat, i); it; ++it, idx++) {
        Yhat(idx) += uid_delta * Vmean(d, it.row());
      }
    }
  }
}

template<class FType>
void MacauPriorVB<FType>::update_beta(Eigen::MatrixXd &Umean) {
  // updating beta and beta_var
  const int nfeat = beta.cols();
  const int blocksize = 4;
  const int num_latent = Umean.rows();

  VectorXd E_lambda      = getElambda( Umean.cols() );
  // E[a_d] - precision for every dimension
  VectorXd E_lambda_beta = lambda_beta_a.cwiseQuotient(lambda_beta_b);

  MatrixXd Z;
  // just in case
  update_uhat();

#pragma omp parallel for private(Z) schedule(static, 1)
  for (int dstart = 0; dstart < num_latent; dstart += blocksize) {
    const int dcount = std::min(blocksize, num_latent - dstart);
    Z.resize(dcount, Umean.cols());

    for (int i = 0; i < Umean.cols(); i++) {
      for (int d = 0; d < dcount; d++) {
        int dx = d + dstart;
        Z(d, i) = Umean(dx, i) - mu_mean(dx) - Uhat(dx, i);
      }
    }

    for (int f = 0; f < nfeat; f++) {
      VectorXd zx(dcount), delta_beta(dcount);
      // zx = Z[dstart : dstart + dcount, :] * F[:, f]
      At_mul_Bt(zx, *F, f, Z);

      for (int d = 0; d < dcount; d++) {
        int dx = d + dstart;
        double A_df     = E_lambda_beta(dx) + E_lambda(dx) * F_colsq(f);
        double B_df     = E_lambda(dx) * (zx(d) + beta(dx,f) * F_colsq(f));
        double A_inv    = 1.0 / A_df;
        double beta_new = B_df * A_inv;
        delta_beta(d)   = beta(dx,f) - beta_new;

        beta_var(dx, f) = A_inv;
        beta(dx, f)     = beta_new;
      }
      // Z[dstart : dstart + dcount, :] += F[:, f] * delta_beta'
      add_Acol_mul_bt(Z, *F, f, delta_beta);
    }
  }
  Uhat_valid = false;
}

template<class FType>
void MacauPriorVB<FType>::update_uhat() {
  if ( ! Uhat_valid) {
    compute_uhat(Uhat, *F, beta);
    Uhat_valid = true;
  }
}

template<class FType>
void MacauPriorVB<FType>::update_prior(Eigen::MatrixXd &Umean, Eigen::MatrixXd &Uvar) {
  // TODO: parallelize or turn on Eigen's parallelism
  assert(Umean.rows() == Uvar.rows());
  assert(Vmean.rows() == Vvar.rows());
  const int N = Umean.cols();

  // Uhat = F * beta
  update_uhat();

  // updating mu_d
  VectorXd Elambda = getElambda(N);
  VectorXd A = Elambda * (b0 + N);
  VectorXd B = Elambda.cwiseProduct( (Umean - Uhat).rowwise().sum() );
  mu_mean = B.cwiseQuotient(A);
 
  for (int i = 0; i < A.size(); i++) {
    mu_var(i) = 1.0 / A(i);
  }

  // updating lambda_b
  lambda_b.setConstant(lambda_b0);
  // += 0.5 * b0 E[mu_d^2]
  lambda_b += 0.5 * b0 * (mu_mean.cwiseProduct(mu_mean) + mu_var);
  // += 0.5 * sum_i (E[uid] - E[mu_d] - E[beta_d]xi)^2
  auto udiff = (Umean - Uhat).colwise() - mu_mean;
  lambda_b += 0.5 * udiff.cwiseProduct(udiff).rowwise().sum();
  // += 0.5 * sum_i (Var[uid])
  lambda_b += 0.5 * Uvar.rowwise().sum();
  // += 0.5 * sum_i (Var[mu_d])
  lambda_b += 0.5 * mu_var * N;
  // += 0.5 * sum_i sum_f Var[beta_df] * x_if * x_if)
  lambda_b += 0.5 * beta_var * F_colsq;

  update_beta(Umean);

  update_lambda_beta();
}

template<class FType>
void MacauPriorVB<FType>::update_lambda_beta() {
  lambda_beta_a = VectorXd::Constant(beta.rows(), lambda_beta_a0 + beta.cols() / 2.0);
  lambda_beta_b = VectorXd::Constant(beta.rows(), lambda_beta_b0);
  const int D = beta.rows();
#pragma omp parallel
  {
    VectorXd tmp(D);
    tmp.setZero();
#pragma omp for schedule(static)
    for (int i = 0; i < beta.cols(); i++) {
      for (int d = 0; d < D; d++) {
        tmp(d) += beta(d, i) * beta(d, i) + beta_var(d, i);
      }
    }
#pragma omp critical
    {
      lambda_beta_b += tmp / 2;
    }
  }
}

template<class FType>
Eigen::VectorXd MacauPriorVB<FType>::getElambda(int N) {
  double lambda_a  = lambda_a0 + (N + 1.0) / 2.0;
  return VectorXd::Constant(lambda_b.size(), lambda_a).cwiseQuotient( lambda_b );
}

template<class FType>
double MacauPriorVB<FType>::getLinkNorm() {
  return beta.norm();
}

template class MacauPriorVB<SparseFeat>;
template class MacauPriorVB<SparseDoubleFeat>;
//template class MacauPriorVB<Eigen::MatrixXd>;
