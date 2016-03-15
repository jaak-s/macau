#include <lapacke.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>

const char lower = 'L';

void chol_decomp(double* A, int n) {
  int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, lower, n, A, n);
  if(info != 0){ throw std::runtime_error("c++ error: Cholesky decomp failed"); }
}

void chol_decomp(Eigen::MatrixXd & A) {
  int info = LAPACKE_dpotrf(LAPACK_COL_MAJOR, lower, A.rows(), A.data(), A.rows());
  if(info != 0){ throw std::runtime_error("c++ error: Cholesky decomp failed (for eigen matrix)"); }
}

void chol_solve(double* A, int n, double* B, int nrhs) {
  int info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, lower, n, nrhs, A, n, B, n);
  if(info != 0){ throw std::runtime_error("c++ error: Cholesky solve failed");}
}

void chol_solve(Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  if (A.rows() != B.rows()) {throw std::runtime_error("A.rows() must equal B.rows()");}
  // TODO: check if B.rows() at the end is a bug (should be B.cols()):
  int info = LAPACKE_dpotrs(LAPACK_COL_MAJOR, lower, A.rows(), B.cols(), A.data(), A.rows(), B.data(), B.rows());
  if(info != 0){ throw std::runtime_error("c++ error: Cholesky solve failed (for eigen matrix)");}
}

/** solves A * X' = B' for X in place */
void chol_solve_t(Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  if (A.rows() != B.cols()) {throw std::runtime_error("A.rows() must equal B.cols()");}
  int info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', A.rows(), B.rows(), A.data(), A.rows(), B.data(), B.rows());
  if(info != 0){ throw std::runtime_error("c++ error: Cholesky solve (chol_solve_t) failed (for eigen matrix)"); }
}

