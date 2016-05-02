#ifndef LATENTPRIORVB_H
#define LATENTPRIORVB_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "linop.h"

/** interface */
class ILatentPriorVB {
  public:
    virtual void update_latents(
        Eigen::MatrixXd &Umean,
        Eigen::MatrixXd &Uvar,
        const Eigen::SparseMatrix<double> &Y,
        const double mean_value,
        const Eigen::MatrixXd &Vmean,
        const Eigen::MatrixXd &Vvar,
        const double alpha) = 0;
    virtual void update_prior(Eigen::MatrixXd &Umean, Eigen::MatrixXd &Uvar) = 0;
    virtual double getLinkNorm() { return NAN; };
    virtual double getLinkLambdaNorm() { return NAN; };
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
    BPMFPriorVB(const int nlatent, const double usquares) { init(nlatent, usquares); }
    BPMFPriorVB() : BPMFPriorVB(10, 1.0) {}

    void init(const int num_latent, const double u_expected_squares);
    void update_latents(
        Eigen::MatrixXd &Umean,
        Eigen::MatrixXd &Uvar,
        const Eigen::SparseMatrix<double> &Y,
        const double mean_value,
        const Eigen::MatrixXd &Vmean,
        const Eigen::MatrixXd &Vvar,
        const double alpha) override;
    void update_prior(Eigen::MatrixXd &Umean, Eigen::MatrixXd &Uvar) override;
    Eigen::VectorXd getElambda(int N);
};

/** Prior with side information (MacauVB) */
template<class FType>
class MacauPriorVB : public ILatentPriorVB {
  public:
    Eigen::VectorXd mu_mean, mu_var; 
    Eigen::VectorXd lambda_b;
    double lambda_a0, lambda_b0;   // Hyper-prior for lambda-s
    double b0;                     // Hyper-prior for Normal-Gamma prior for mu_mean (2.0)

    Eigen::MatrixXd Uhat;
    std::unique_ptr<FType> F;  // side information
    Eigen::VectorXd F_colsq;   // sum-of-squares for every feature (column)
    Eigen::MatrixXd beta;      // link matrix
    Eigen::MatrixXd beta_var;  // link matrix variance

    Eigen::VectorXd lambda_beta_a;
    Eigen::VectorXd lambda_beta_b;
    double lambda_beta_a0;     // Hyper-prior for lambda_beta
    double lambda_beta_b0;     // Hyper-prior for lambda_beta

  public:
    MacauPriorVB(const int nlatent, std::unique_ptr<FType> & Fmat, double usquares) { init(nlatent, Fmat, usquares); }
    MacauPriorVB() {}

    void init(const int num_latent, std::unique_ptr<FType> & Fmat, double usquares);

    void update_latents(
        Eigen::MatrixXd &Umean,
        Eigen::MatrixXd &Uvar,
        const Eigen::SparseMatrix<double> &Y,
        const double mean_value,
        const Eigen::MatrixXd &Vmean,
        const Eigen::MatrixXd &Vvar,
        const double alpha) override;
    void update_prior(Eigen::MatrixXd &Umean, Eigen::MatrixXd &Uvar) override;
    void update_beta(const Eigen::MatrixXd &U);
    void update_lambda_beta();
    double getLinkNorm() override;
    //double getLinkLambdaNorm() override { return lambda_beta; };
    Eigen::VectorXd getElambda(int N);
};
#endif /* LATENTPRIORVB_H */
