#ifndef LATENTPRIOR_H
#define LATENTPRIOR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "mvnormal.h"

/** interface */
class ILatentPrior {
  public:
    void virtual sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                        const Eigen::MatrixXd &samples, double alpha, const int num_latent) {};
    void virtual update_prior(const Eigen::MatrixXd &U) {};
};

/** Prior without side information (pure BPMF) */
class BPMFPrior : public ILatentPrior {
  public:
    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    int b0;
    int df;

  public:
    BPMFPrior(const int nlatent) { init(nlatent); }
    BPMFPrior() : BPMFPrior(10) {}

    void init(const int num_latent) {
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

    void sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                                   const Eigen::MatrixXd &samples, double alpha, const int num_latent);
    void update_prior(const Eigen::MatrixXd &U);
};

/** Prior without side information (pure BPMF) */
class MacauPrior : public ILatentPrior {
  private:
    Eigen::MatrixXd Uhat;
    Eigen::MatrixXd F;    /* side information */
    Eigen::MatrixXd FtF;  /* F'F */
    Eigen::MatrixXd beta; /* link matrix */
    bool use_FtF;
    double lambda_beta;

    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    int b0;
    int df;

    double tol = 1e-6;

  public:
    MacauPrior(const int nlatent, Eigen::MatrixXd & Fmat, bool comp_FtF) { init(nlatent, Fmat, comp_FtF); }
    MacauPrior() {}

    void init(const int num_latent, Eigen::MatrixXd & Fmat, bool comp_FtF);

    void sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                                   const Eigen::MatrixXd &samples, double alpha, const int num_latent);
    void update_prior(const Eigen::MatrixXd &U);
    void sample_beta(const Eigen::MatrixXd &U);
};

void sample_latent(Eigen::MatrixXd &s, int mm, const Eigen::SparseMatrix<double> &mat, double mean_rating,
    const Eigen::MatrixXd &samples, double alpha, const Eigen::VectorXd &mu_u, const Eigen::MatrixXd &Lambda_u,
    const int num_latent);
void sample_latent_blas(Eigen::MatrixXd &s, int mm, const Eigen::SparseMatrix<double> &mat, double mean_rating,
    const Eigen::MatrixXd &samples, double alpha, const Eigen::VectorXd &mu_u, const Eigen::MatrixXd &Lambda_u,
    const int num_latent);

template<typename T>
void At_mul_A(Eigen::MatrixXd & result, const T & F);
void At_mul_A(Eigen::MatrixXd & result, const Eigen::MatrixXd & F);

template<typename T>
Eigen::MatrixXd A_mul_B(const Eigen::MatrixXd & A, const T & B);
Eigen::MatrixXd A_mul_B(const Eigen::MatrixXd & A, const Eigen::MatrixXd & B);

#endif /* LATENTPRIOR_H */
