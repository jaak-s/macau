#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "latentprior.h"

template<class FType>
class MacauOnePrior : public ILatentPrior {
  public:
    int num_latent;
    Eigen::MatrixXd Uhat;

    std::unique_ptr<FType> F;  /* side information */
    Eigen::VectorXd F_colsq;   // sum-of-squares for every feature (column)

    Eigen::MatrixXd beta;      /* link matrix */
    Eigen::VectorXd lambda_beta;
    double lambda_beta_a0; /* Hyper-prior for lambda_beta */
    double lambda_beta_b0; /* Hyper-prior for lambda_beta */

    Eigen::VectorXd mu; 
    Eigen::VectorXd lambda;
    double lambda_a0;
    double lambda_b0;

    int l0;

  public:
    MacauOnePrior(const int nlatent, std::unique_ptr<FType> &Fmat) { init(nlatent, Fmat); }
    MacauOnePrior() {}

    void init(const int nlatent, std::unique_ptr<FType> &Fmat);

    void sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                                   const Eigen::MatrixXd &samples, double alpha, const int num_latent) override;
    void sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                   double mean_value, const Eigen::MatrixXd &samples, const int num_latent) override;
    void sample_latents(ProbitNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) override;
    void sample_latents(double noisePrecision, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) override;
    void update_prior(const Eigen::MatrixXd &U) override;
    double getLinkNorm() override { return beta.norm(); };
    double getLinkLambda() { return lambda_beta.mean(); };
    void sample_beta(const Eigen::MatrixXd &U);
    void sample_mu_lambda(const Eigen::MatrixXd &U);
    void sample_lambda_beta();
    void setLambdaBeta(double lb) { lambda_beta = Eigen::VectorXd::Constant(num_latent, lb); };
    void saveModel(std::string prefix) override;
};

