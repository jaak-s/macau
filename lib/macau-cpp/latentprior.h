#ifndef LATENTPRIOR_H
#define LATENTPRIOR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "mvnormal.h"
#include "linop.h"

/** interface */
class ILatentPrior {
  public:
    void virtual sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                        const Eigen::MatrixXd &samples, double alpha, const int num_latent) {};
    void virtual update_prior(const Eigen::MatrixXd &U) {};
    virtual double getLinkNorm() { return NAN; };
    virtual double getLinkLambda() { return NAN; };
    virtual ~ILatentPrior() {};
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
    void init(const int num_latent);

    void sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                                   const Eigen::MatrixXd &samples, double alpha, const int num_latent);
    void update_prior(const Eigen::MatrixXd &U);
};

/** Prior without side information (pure BPMF) */
template<class FType>
class MacauPrior : public ILatentPrior {
  private:
    Eigen::MatrixXd Uhat;
    std::unique_ptr<FType> F;  /* side information */
    Eigen::MatrixXd FtF;       /* F'F */
    Eigen::MatrixXd beta;      /* link matrix */
    bool use_FtF;
    double lambda_beta;
    double lambda_beta_mu0; /* Hyper-prior for lambda_beta */
    double lambda_beta_nu0; /* Hyper-prior for lambda_beta */

    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    int b0;
    int df;

    double tol = 1e-6;

  public:
    MacauPrior(const int nlatent, std::unique_ptr<FType> &Fmat, bool comp_FtF) { init(nlatent, Fmat, comp_FtF); }
    MacauPrior() {}

    void init(const int num_latent, std::unique_ptr<FType> &Fmat, bool comp_FtF);

    void sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                                   const Eigen::MatrixXd &samples, double alpha, const int num_latent);
    void update_prior(const Eigen::MatrixXd &U);
    double getLinkNorm();
    double getLinkLambda() { return lambda_beta; };
    void sample_beta(const Eigen::MatrixXd &U);
    void setLambdaBeta(double lambda_beta);
};

std::pair<double,double> posterior_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);
double sample_lambda_beta(Eigen::MatrixXd & beta, Eigen::MatrixXd & Lambda_u, double nu, double mu);

void sample_latent(Eigen::MatrixXd &s,
                   int mm,
                   const Eigen::SparseMatrix<double> &mat,
                   double mean_rating,
                   const Eigen::MatrixXd &samples,
                   double alpha,
                   const Eigen::VectorXd &mu_u,
                   const Eigen::MatrixXd &Lambda_u,
                   const int num_latent);

void sample_latent_blas(Eigen::MatrixXd &s,
                        int mm,
                        const Eigen::SparseMatrix<double> &mat,
                        double mean_rating,
                        const Eigen::MatrixXd &samples,
                        double alpha,
                        const Eigen::VectorXd &mu_u,
                        const Eigen::MatrixXd &Lambda_u,
                        const int num_latent);

Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, Eigen::MatrixXd & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseFeat & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseDoubleFeat & B);

#endif /* LATENTPRIOR_H */
