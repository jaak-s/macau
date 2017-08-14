#ifndef MVNORMAL_H
#define MVNORMAL_H

#define EIGEN_RUNTIME_NO_MALLOC
//#define EIGEN_DONT_PARALLELIZE 1

#include <Eigen/Dense>                                                                                                 
#include <Eigen/Sparse>
#include <random>

double randn0();
double randn(double);

void bmrandn(double* x, long n);
void bmrandn(Eigen::MatrixXd & X);
double bmrandn_single();
void bmrandn_single(double* x, long n);
void bmrandn_single(Eigen::VectorXd & x);
void init_bmrng();
void init_bmrng(int seed);

double rand_unif();
double rand_unif(double low, double high);

double rgamma(double shape, double scale);

auto nrandn(int n) -> decltype( Eigen::VectorXd::NullaryExpr(n, std::cref(randn)) ); 

std::pair<Eigen::VectorXd, Eigen::MatrixXd> CondNormalWishart(const Eigen::MatrixXd &U, const Eigen::VectorXd &mu, const double kappa, const Eigen::MatrixXd &T, const int nu);

Eigen::MatrixXd MvNormal_prec(const Eigen::MatrixXd & Lambda, int nn);
Eigen::MatrixXd MvNormal_prec(const Eigen::MatrixXd & Lambda, const Eigen::VectorXd & mean, int nn);
Eigen::MatrixXd MvNormal_prec_omp(const Eigen::MatrixXd & Lambda, int nn);
Eigen::MatrixXd MvNormal(const Eigen::MatrixXd covar, const Eigen::VectorXd mean, int nn);

#endif /* MVNORMAL_H */
