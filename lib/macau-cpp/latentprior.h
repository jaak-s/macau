#ifndef LATENTPRIOR_H
#define LATENTPRIOR_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "macau.h"
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
  private:
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


#endif /* LATENTPRIOR_H */
