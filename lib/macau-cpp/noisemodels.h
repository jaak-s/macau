#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iomanip>

#include "latentprior.h"
#include "sparsetensor.h"
#include "bpmfutils.h"

/** interface */
class INoiseModel {
  public:
    virtual void init(MatrixData & data) { };
    virtual void init(TensorData & data) { };

    virtual void update(MatrixData & data, std::vector< std::unique_ptr<Eigen::MatrixXd> > &samples) { };
    virtual void update(TensorData & data, std::vector< std::unique_ptr<Eigen::MatrixXd> > &samples) { };

    virtual void evalModel(MatrixData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) = 0;
    virtual void evalModel(TensorData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) = 0;

    virtual double getEvalMetric() = 0;
    virtual std::string getEvalString() = 0;
    virtual std::string getInitStatus() = 0;
    virtual std::string getStatus() = 0;
    virtual ~INoiseModel() {};
};

/** Gaussian noise is fixed for the whole run */
class FixedGaussianNoise : public INoiseModel {
  public:
    double alpha;
    double rmse_test = NAN;
    double rmse_test_onesample = NAN;
  
    FixedGaussianNoise(double a) { alpha = a; }
    FixedGaussianNoise() : FixedGaussianNoise(1.0) {};

    std::string getInitStatus() override { return std::string("Noise precision: ") + std::to_string(alpha) + " (fixed)"; }
    std::string getStatus() override { return std::string(""); }

    void setPrecision(double a) { alpha = a; }    
    double getEvalMetric() override { return rmse_test;}
    std::string getEvalString() { return std::string("RMSE: ") + to_string_with_precision(rmse_test,5) + " (1samp: " + to_string_with_precision(rmse_test_onesample,5)+")";}
 
    void evalModel(MatrixData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) override;
    void evalModel(TensorData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) override;
};

/** Gaussian noise that adapts to the data */
class AdaptiveGaussianNoise : public INoiseModel {
  public:
    double alpha = NAN;
    double alpha_max = NAN;
    double sn_max;
    double sn_init;
    double var_total = NAN;
    double rmse_test = NAN;
    double rmse_test_onesample = NAN;

    AdaptiveGaussianNoise(double sinit, double smax) {
      sn_max  = smax;
      sn_init = sinit;
    }
    AdaptiveGaussianNoise() : AdaptiveGaussianNoise(1.0, 10.0) {}

    void init(MatrixData & data) override;
    void init(TensorData & data) override;
    void update(MatrixData & data, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) override;
    void update(TensorData & data, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) override;
    void setSNInit(double a) { sn_init = a; }
    void setSNMax(double a)  { sn_max  = a; }
    std::string getInitStatus() override { return std::string("Noise precision: adaptive (with max precision of ") + std::to_string(alpha_max) + ")"; }

    std::string getStatus() override {
      return std::string("Prec:") + to_string_with_precision(alpha, 2);
    }

    double getEvalMetric() override {return rmse_test;}
    std::string getEvalString() { return std::string("RMSE: ") + to_string_with_precision(rmse_test,5) + " (1samp: " + to_string_with_precision(rmse_test_onesample,5)+")";}

    void evalModel(MatrixData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) override;
    void evalModel(TensorData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) override;
};

/** Probit noise model (binary). Fixed for the whole run */
class ProbitNoise : public INoiseModel {
  public:
    double auc_test = NAN;
    double auc_test_onesample = NAN;
    ProbitNoise() { }

    std::string getInitStatus() override { return std::string("Probit noise model"); }
    std::string getStatus() override { return std::string(""); }
    double getEvalMetric() override {return auc_test;}
    std::string getEvalString() { return std::string("AUC: ") + to_string_with_precision(auc_test,5) + " (1samp: " + to_string_with_precision(auc_test_onesample,5)+")";}

    void evalModel(MatrixData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) override;
    void evalModel(TensorData & data, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) override;
};
