#ifndef MACAU_H
#define MACAU_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

class Macau {
  int num_latent;

  double alpha = 2.0; 
  int nsamples = 100;
  int burnin   = 50;

  double mean_rating = .0; 
  Eigen::SparseMatrix<double> Y, Yt, Ytest;

  double rmse_test  = .0;
  double rmse_train = .0;

  /** BPMF model */
  Eigen::VectorXd mu_u; 
  Eigen::VectorXd mu_m;
  Eigen::MatrixXd Lambda_u;
  Eigen::MatrixXd Lambda_m;
  Eigen::MatrixXd sample_u;
  Eigen::MatrixXd sample_m;

  /** Hyper-parameters, Inv-Wishart */
  Eigen::MatrixXd WI_u, WI_m;
  Eigen::VectorXd mu0_u, mu0_m;

  int b0_u = 2;
  int b0_m = 2;

  int df_u;
  int df_m;

  public:
    Macau(int D) : num_latent{D}, df_u{D}, df_m{D} {}
    Macau() : Macau(10) {}
    void setPrecision(double p);
    void setSamples(int burnin, int nsamples);
    void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols);
    void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols);
    double getRmseTest();
    void init();
    void run();
};

void sparseFromIJV(Eigen::SparseMatrix<double> & X, int* rows, int* cols, double* values, int N);

std::pair<double,double> eval_rmse(Eigen::SparseMatrix<double> & P, int n, Eigen::VectorXd & predictions, const Eigen::MatrixXd &sample_m, const Eigen::MatrixXd &sample_u, double mean_rating);

void sample_latent(Eigen::MatrixXd &s, int mm, const Eigen::SparseMatrix<double> &mat, double mean_rating,
    const Eigen::MatrixXd &samples, double alpha, const Eigen::VectorXd &mu_u, const Eigen::MatrixXd &Lambda_u,
    const int num_latent);
#endif /* MACAU_H */
