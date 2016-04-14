#ifndef LATENTPRIORVB_H
#define LATENTPRIORVB_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "linop.h"

/** interface */
class ILatentPriorVB {
  public:
    void virtual update_latents(
        Eigen::MatrixXd &Umean,
        Eigen::MatrixXd &Uvar,
        const Eigen::SparseMatrix<double> &Y,
        const double mean_value,
        const Eigen::MatrixXd &Vmean,
        const Eigen::MatrixXd &Vvar,
        const double alpha) {};
    void virtual update_prior(const Eigen::MatrixXd &Umean, const Eigen::MatrixXd &Uvar) {};
    virtual double getLinkNorm() { return NAN; };
    virtual double getLinkLambda() { return NAN; };
    virtual ~ILatentPriorVB() {};
};

/** Prior without side information (pure BPMF) */
class BPMFPriorVB : public ILatentPriorVB {
  public:
    Eigen::VectorXd mu_mean, mu_var; 
    Eigen::VectorXd lambda_b;
    double lambda_a0, lambda_b0;   // Hyper-prior for lambda-s
    double b0;                     // Hyper-prior for Normal-Gamma prior for mu_mean (2.0)

  public:
    BPMFPriorVB(const int nlatent) { init(nlatent); }
    BPMFPriorVB() : BPMFPriorVB(10) {}

    void init(const int num_latent);
    void update_latents(
        Eigen::MatrixXd &Umean,
        Eigen::MatrixXd &Uvar,
        const Eigen::SparseMatrix<double> &Y,
        const double mean_value,
        const Eigen::MatrixXd &Vmean,
        const Eigen::MatrixXd &Vvar,
        const double alpha);
    void update_prior(const Eigen::MatrixXd &Umean, const Eigen::MatrixXd &Uvar);
    Eigen::VectorXd getElambda(int N);
};

/** Prior without side information (pure BPMF) */
/*
template<class FType>
class MacauPriorVB : public ILatentPriorVB {
  private:
    Eigen::MatrixXd Uhat;
    std::unique_ptr<FType> F;  // side information
    Eigen::MatrixXd FtF;       // F'F
    Eigen::MatrixXd beta;      // link matrix
    bool use_FtF;
    double lambda_beta;
    double lambda_beta_mu0; // Hyper-prior for lambda_beta
    double lambda_beta_nu0; // Hyper-prior for lambda_beta

    Eigen::VectorXd mu; 
    Eigen::MatrixXd Lambda;
    Eigen::MatrixXd WI;
    Eigen::VectorXd mu0;

    int b0;
    int df;

    double tol = 1e-6;

  public:
    MacauPriorVB(const int nlatent, FType * Fmat, bool comp_FtF) { init(nlatent, Fmat, comp_FtF); }
    MacauPriorVB() {}

    void init(const int num_latent, FType * Fmat, bool comp_FtF);

    void sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                                   const Eigen::MatrixXd &samples, double alpha, const int num_latent);
    void update_prior(const Eigen::MatrixXd &U);
    double getLinkNorm();
    double getLinkLambda() { return lambda_beta; };
    void sample_beta(const Eigen::MatrixXd &U);
    void setLambdaBeta(double lambda_beta);
};
*/
#endif /* LATENTPRIORVB_H */
