#ifndef MACAU_H
#define MACAU_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <memory>
#include "latentprior.h"

// try adding num_latent as template parameter to Macau
class Macau {
  public:
  int num_latent;

  double alpha = 2.0; 
  int nsamples = 100;
  int burnin   = 50;

  double mean_rating = .0; 
  Eigen::SparseMatrix<double> Y, Yt, Ytest;
  Eigen::VectorXd predictions;
  Eigen::VectorXd predictions_var;

  double rmse_test  = .0;
  double rmse_train = .0;

  /** BPMF model */
  std::vector< std::unique_ptr<ILatentPrior> > priors;
  std::vector< std::unique_ptr<Eigen::MatrixXd> > samples;
  bool verbose = true;

  public:
    Macau(int D) : num_latent{D} {}
    Macau() : Macau(10) {}
    void addPrior(std::unique_ptr<ILatentPrior> & prior);
    void setPrecision(double p);
    void setSamples(int burnin, int nsamples);
    void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols);
    void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols);
    void init();
    void run();
    void printStatus(int i, double rmse, double rmse_avg, double elapsedi, double samples_per_sec);
    void setVerbose(bool v) { verbose = v; };
    double getRmseTest();
    Eigen::VectorXd getPredictions() { return predictions; };
    Eigen::VectorXd getStds();
    Eigen::MatrixXd getTestData();
    ~Macau();
};

void sparseFromIJV(Eigen::SparseMatrix<double> & X, int* rows, int* cols, double* values, int N);

#endif /* MACAU_H */
