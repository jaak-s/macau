#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <memory>
#include "latentprior.h"

// try adding num_latent as template parameter to Macau
class Macau {
  public:
  int num_latent;

  //double alpha = 2.0; 
  std::unique_ptr<INoiseModel> noise;
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

  bool save_model = false;
  std::string save_prefix = "model";

  public:
    Macau(int D) : num_latent{D} {}
    Macau() : Macau(10) {}
    void addPrior(std::unique_ptr<ILatentPrior> & prior);
    void setPrecision(double p);
    void setAdaptivePrecision(double sn_init, double sn_max);
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
    void saveModel(int isample);
    void saveGlobalParams();
    void setSaveModel(bool save) { save_model = save; };
    void setSavePrefix(std::string pref) { save_prefix = pref; };
    ~Macau();
};

std::pair<double,double> eval_rmse(Eigen::SparseMatrix<double> & P, int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var, const Eigen::MatrixXd &sample_m, const Eigen::MatrixXd &sample_u, double mean_rating);
void sparseFromIJV(Eigen::SparseMatrix<double> & X, int* rows, int* cols, double* values, int N);

