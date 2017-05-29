#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include "mvnormal.h"
#include "linop.h"
#include "sparsetensor.h"

 // forward declarations
class FixedGaussianNoise;
class AdaptiveGaussianNoise;
class ProbitNoise;
class MatrixData;

/** interface */
class ILatentPrior {
  public:
    virtual void sample_latents(Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat, double mean_value,
                        const Eigen::MatrixXd &samples, double alpha, const int num_latent) = 0;
    virtual void sample_latents(FixedGaussianNoise& noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                        double mean_value, const Eigen::MatrixXd &samples, const int num_latent);
    virtual void sample_latents(AdaptiveGaussianNoise& noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                        double mean_value, const Eigen::MatrixXd &samples, const int num_latent);
    virtual void sample_latents(ProbitNoise& noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                        double mean_value, const Eigen::MatrixXd &samples, const int num_latent) = 0;
    // general functions (called from outside)
    void sample_latents(FixedGaussianNoise& noiseModel, MatrixData & matrixData,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    void sample_latents(AdaptiveGaussianNoise& noiseModel, MatrixData & matrixData,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    void sample_latents(ProbitNoise& noiseModel, MatrixData & matrixData,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    // for tensor
    void sample_latents(FixedGaussianNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    void sample_latents(AdaptiveGaussianNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent);
    virtual void sample_latents(ProbitNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) = 0;
    virtual void sample_latents(double noisePrecision, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) = 0;

    void virtual update_prior(const Eigen::MatrixXd &U) {};
    virtual double getLinkNorm() { return NAN; };
    virtual double getLinkLambda() { return NAN; };
    virtual void saveModel(std::string prefix) {};
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
                                   const Eigen::MatrixXd &samples, double alpha, const int num_latent) override;
    void sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                   double mean_value, const Eigen::MatrixXd &samples, const int num_latent) override;

    void sample_latents(ProbitNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) override;
    void sample_latents(double noisePrecision, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) override;
    void update_prior(const Eigen::MatrixXd &U) override;
    void saveModel(std::string prefix) override;
};

/** Prior without side information (pure BPMF) */
template<class FType>
class MacauPrior : public ILatentPrior {
  public:
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
                                   const Eigen::MatrixXd &samples, double alpha, const int num_latent) override;
    void sample_latents(ProbitNoise & noise, Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                   double mean_value, const Eigen::MatrixXd &samples, const int num_latent) override;
    void sample_latents(ProbitNoise& noiseModel, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) override;
    void sample_latents(double noisePrecision, TensorData & data,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples, int mode, const int num_latent) override;
    void update_prior(const Eigen::MatrixXd &U) override;
    double getLinkNorm();
    double getLinkLambda() { return lambda_beta; };
    void sample_beta(const Eigen::MatrixXd &U);
    void setLambdaBeta(double lb) { lambda_beta = lb; };
    void setTol(double t) { tol = t; };
    void saveModel(std::string prefix) override;
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

void sample_latent_blas_probit(Eigen::MatrixXd &s,
                        int mm,
                        const Eigen::SparseMatrix<double> &mat,
                        double mean_rating,
                        const Eigen::MatrixXd &samples,
                        const Eigen::VectorXd &mu_u,
                        const Eigen::MatrixXd &Lambda_u,
                        const int num_latent);
void sample_latent_tensor(std::unique_ptr<Eigen::MatrixXd> &U,
                          int n,
                          std::unique_ptr<SparseMode> & sparseMode,
                          VectorView<Eigen::MatrixXd> & view,
                          double mean_value,
                          double alpha,
                          Eigen::VectorXd & mu,
                          Eigen::MatrixXd & Lambda);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, Eigen::MatrixXd & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseFeat & B);
Eigen::MatrixXd A_mul_B(Eigen::MatrixXd & A, SparseDoubleFeat & B);

MacauPrior<Eigen::MatrixXd>* make_dense_prior(int nlatent, double* ptr, int nrows, int ncols, bool colMajor, bool comp_FtF);
