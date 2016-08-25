#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <iomanip>

#include "latentprior.h"

class ILatentPrior; // forward declaration

/** interface */
class INoiseModel {
  public:
    virtual void sample_latents(std::unique_ptr<ILatentPrior> & lprior,
                                Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                                double mean_value, const Eigen::MatrixXd &samples, const int num_latent) = 0;
    virtual void init(const Eigen::SparseMatrix<double> &train, double mean_value) = 0;
    virtual void update(const Eigen::SparseMatrix<double> &train, double mean_value, std::vector< std::unique_ptr<Eigen::MatrixXd> > &samples) = 0;
    virtual std::string getInitStatus() = 0;
    virtual std::string getStatus() = 0;
    virtual ~INoiseModel() {};
};

template<class T>
class INoiseModelDisp : public INoiseModel {
    void sample_latents(std::unique_ptr<ILatentPrior> & lprior,
                              Eigen::MatrixXd &U, const Eigen::SparseMatrix<double> &mat,
                              double mean_value, const Eigen::MatrixXd &samples, const int num_latent) override;
};

/** Gaussian noise is fixed for the whole run */
class FixedGaussianNoise : public INoiseModelDisp<FixedGaussianNoise> {
  public:
    double alpha;
  
    FixedGaussianNoise(double a) { alpha = a; }
    FixedGaussianNoise() : FixedGaussianNoise(1.0) {};

    void init(const Eigen::SparseMatrix<double> &train, double mean_value) override { }
    void update(const Eigen::SparseMatrix<double> &train, double mean_value, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) override {}
    std::string getInitStatus() override { return std::string("Noise precision: ") + std::to_string(alpha) + " (fixed)"; }
    std::string getStatus() override { return std::string(""); }

    void setPrecision(double a) { alpha = a; }
};

/** Gaussian noise that adapts to the data */
class AdaptiveGaussianNoise : public INoiseModelDisp<AdaptiveGaussianNoise> {
  public:
    double alpha = NAN;
    double alpha_max = NAN;
    double sn_max;
    double sn_init;
    double var_total = NAN;

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
};
