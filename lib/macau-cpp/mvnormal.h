#ifndef MVNORMAL_H
#define MVNORMAL_H

#define EIGEN_RUNTIME_NO_MALLOC
//#define EIGEN_DONT_PARALLELIZE 1

#include <Eigen/Dense>                                                                                                 
#include <Eigen/Sparse>

double randn0();
double randn(double);

auto nrandn(int n) -> decltype( Eigen::VectorXd::NullaryExpr(n, std::cref(randn)) ); 

std::pair<Eigen::VectorXd, Eigen::MatrixXd> CondNormalWishart(const Eigen::MatrixXd &U, const Eigen::VectorXd &mu, const double kappa, const Eigen::MatrixXd &T, const int nu);

#endif /* MVNORMAL_H */
