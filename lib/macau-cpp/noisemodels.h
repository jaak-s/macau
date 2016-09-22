#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iomanip>

#include "latentprior.h"
#include "sparsetensor.h"

// forward declarations
class ILatentPrior;
class IData;

/** interface */
class INoiseModel {
  public:
    virtual void sample_latents(std::unique_ptr<ILatentPrior> & lprior,
                                Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                double mean_value, const Eigen::MatrixXd &samples, const int num_latent) = 0;
    virtual void sample_latents(std::unique_ptr<ILatentPrior> & prior,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                                std::unique_ptr<IData> & data,
                                int mode,
                                const int num_latent) = 0;
    virtual void init(const Eigen::SparseMatrix<double> &train, double mean_value) = 0;
    virtual void update(const Eigen::SparseMatrix<double> &train, double mean_value, std::vector< std::unique_ptr<Eigen::MatrixXd> > &samples) = 0;
    virtual void evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, const Eigen::MatrixXd &cols, const Eigen::MatrixXd &rows, double mean_rating) = 0;
    virtual double getEvalMetric() = 0;
    virtual std::string getEvalString() = 0;
    virtual std::string getInitStatus() = 0;
    virtual std::string getStatus() = 0;
    virtual ~INoiseModel() {};
};

template<class T>
class INoiseModelDisp : public INoiseModel {
    void sample_latents(std::unique_ptr<ILatentPrior> & lprior,
                        Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                        double mean_value, const Eigen::MatrixXd &samples, const int num_latent) override;
    void sample_latents(std::unique_ptr<ILatentPrior> & prior,
                        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                        std::unique_ptr<IData> & data,
                        int mode,
                        const int num_latent) override;
};

/** Gaussian noise is fixed for the whole run */
class FixedGaussianNoise : public INoiseModelDisp<FixedGaussianNoise> {
  public:
    double alpha;
    double rmse_test;
    double rmse_test_onesample;
  
    FixedGaussianNoise(double a) { alpha = a; }
    FixedGaussianNoise() : FixedGaussianNoise(1.0) {};

    void init(const Eigen::SparseMatrix<double> &train, double mean_value) override { }
    void update(const Eigen::SparseMatrix<double> &train, double mean_value, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) override {}
    std::string getInitStatus() override { return std::string("Noise precision: ") + std::to_string(alpha) + " (fixed)"; }
    std::string getStatus() override { return std::string(""); }

    void setPrecision(double a) { alpha = a; }    
    void evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, const Eigen::MatrixXd &cols, const Eigen::MatrixXd &rows, double mean_rating) override;
    double getEvalMetric() override { return rmse_test;}
    std::string getEvalString() { return std::string("RMSE: ") + to_string_with_precision(rmse_test,5) + " (1samp: " + to_string_with_precision(rmse_test_onesample,5)+")";}
 
};

/** Gaussian noise that adapts to the data */
class AdaptiveGaussianNoise : public INoiseModelDisp<AdaptiveGaussianNoise> {
  public:
    double alpha = NAN;
    double alpha_max = NAN;
    double sn_max;
    double sn_init;
    double var_total = NAN;
    double rmse_test;
    double rmse_test_onesample;

    AdaptiveGaussianNoise(double sinit, double smax) {
      sn_max  = smax;
      sn_init = sinit;
    }
    AdaptiveGaussianNoise() : AdaptiveGaussianNoise(1.0, 10.0) {}

    void init(const Eigen::SparseMatrix<double> &train, double mean_value) override;
    void update(const Eigen::SparseMatrix<double> &train, double mean_value, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) override;
    void setSNInit(double a) { sn_init = a; }
    void setSNMax(double a)  { sn_max  = a; }
    std::string getInitStatus() override { return std::string("Noise precision: adaptive (with max precision of ") + std::to_string(alpha_max) + ")"; }

    std::string getStatus() override {
      return std::string("Prec:") + to_string_with_precision(alpha, 2);
    }

    void evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, const Eigen::MatrixXd &cols, const Eigen::MatrixXd &rows, double mean_rating) override;
    double getEvalMetric() override {return rmse_test;}
    std::string getEvalString() { return std::string("RMSE: ") + to_string_with_precision(rmse_test,5) + " (1samp: " + to_string_with_precision(rmse_test_onesample,5)+")";}
};

/** Probit noise model (binary). Fixed for the whole run */
class ProbitNoise : public INoiseModelDisp<ProbitNoise> {
  public:
    double auc_test;
    double auc_test_onesample;
    ProbitNoise() { }

    void init(const Eigen::SparseMatrix<double> &train, double mean_value) override { }
    void update(const Eigen::SparseMatrix<double> &train, double mean_value, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) override {}
    std::string getInitStatus() override { return std::string("Probit noise model"); }
    std::string getStatus() override { return std::string(""); }
    void evalModel(Eigen::SparseMatrix<double> & Ytest, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, const Eigen::MatrixXd &cols, const Eigen::MatrixXd &rows, double mean_rating) override;
    double getEvalMetric() override {return auc_test;}
    std::string getEvalString() { return std::string("AUC: ") + to_string_with_precision(auc_test,5) + " (1samp: " + to_string_with_precision(auc_test_onesample,5)+")";}
};
