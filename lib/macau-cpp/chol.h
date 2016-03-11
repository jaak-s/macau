#ifndef CHOL_H
#define CHOL_H

#include <Eigen/Dense>

void chol_decomp(Eigen::MatrixXd & A);
void chol_decomp(double* A, int n);
void chol_solve(Eigen::MatrixXd & A, Eigen::MatrixXd & B);
void chol_solve(double* A, int n, double* B, int nrhs);
void chol_solve_t(Eigen::MatrixXd & A, Eigen::MatrixXd & B);

#endif /* CHOL_H */
