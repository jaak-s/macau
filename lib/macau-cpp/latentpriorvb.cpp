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

  lambda_b.resize(num_latent);
  lambda_b.setConstant(0.1);

  // parameters of Normal-Gamma distribution
  lambda_a0 = 1.0;
  lambda_b0 = 1.0;
  b0 = 2.0;
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
      Yhat(idx) = Umean.col(i).dot( Vmean.col(it.row()) );
      idx++;
    }
    for (int d = 0; d < D; d++) {
      // computing Lid
      const double uid = Umean(d, i);
      double Lid = Elambda_Emu(d);

      idx = 0;
      for ( SparseMatrix<double>::InnerIterator it(Ymat, i); it; ++it, idx++) {
        const double vjd = Vmean(d, it.row());
        // Lid += alpha * (Yij - E[kijd]) * E[vjd]
        Lid += alpha * (it.value() - (Yhat(idx) - uid)) * vjd;
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

void BPMFPriorVB::update_prior(const Eigen::MatrixXd &Umean, const Eigen::MatrixXd &Uvar) {
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
  // += b0 E[mu_d^2]
  lambda_b += 0.5 * b0 * (mu_mean.cwiseProduct(mu_mean) + mu_var);
  // += sum_i (E[uid] - E[mu_d])^2
  lambda_b += 0.5 * (Umean.colwise() - mu_mean).cwiseProduct(Umean.colwise() - mu_mean).rowwise().sum();
  // += sum_i (Var[uid])
  lambda_b += 0.5 * Uvar.rowwise().sum();
  lambda_b += 0.5 * mu_var * N;
}
