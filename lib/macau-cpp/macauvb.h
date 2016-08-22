#ifndef MACAUVB_H
#define MACAUVB_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <memory>
#include "latentpriorvb.h"

// try adding num_latent as template parameter to Macau
class MacauVB {
  public:
  int num_latent;

  double alpha = 2.0; 
  int niter = 100;
  double latent_init_std = 1.0;

  double mean_value = .0; 
  Eigen::SparseMatrix<double> Y, Yt, Ytest;
  Eigen::VectorXd predictions;

  double rmse_test  = .0;
  double rmse_train = .0;

  /** BPMF model */
  std::vector< std::unique_ptr<ILatentPriorVB> > priors;
  std::vector< std::unique_ptr<Eigen::MatrixXd> > samples_mean;
  std::vector< std::unique_ptr<Eigen::MatrixXd> > samples_var;
  bool verbose = true;

  bool save_model = false;
  std::string save_prefix = "model";

  public:
    MacauVB(int D) : num_latent{D} {};
    MacauVB() : MacauVB(10) {};
    void addPrior(std::unique_ptr<ILatentPriorVB> & prior);
    void setPrecision(double p);
    void setNiter(int niter);
    void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols);
    void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols);
    void init();
    void run();
    void printStatus(int i, double rmse, double elapsedi, double updates_per_sec);
    void setVerbose(bool v) { verbose = v; };
    double getRmseTest() { return rmse_test; };
    Eigen::VectorXd getPredictions() { return predictions; };
    Eigen::VectorXd getStds();
    Eigen::MatrixXd getTestData();
    void saveModel();
    void setSaveModel(bool save) { save_model = save; };
    void setSavePrefix(std::string pref) { save_prefix = pref; };
    ~MacauVB() {};
};

double eval_rmse(Eigen::VectorXd & predictions, const Eigen::SparseMatrix<double> & P, const Eigen::MatrixXd &sample_m, const Eigen::MatrixXd &sample_u, const double mean_value);

#endif /* MACAUVB_H */
