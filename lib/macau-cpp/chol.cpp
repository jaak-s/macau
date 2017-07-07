//#include <lapacke.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <iostream>

char lower = 'L';

// TODO, remove dependency on lapacke.h
extern "C" void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void dpotrs_(char *uplo, int* n, int* nrhs, double* A, int* lda, double* B, int* ldb, int* info);

void chol_decomp(double* A, int n) {
  int info;
  dpotrf_(&lower, &n, A, &n, &info);
  if(info != 0){ throw std::runtime_error("c++ error: Cholesky decomp failed"); }
}

void chol_decomp(Eigen::MatrixXd & A) {
  int info, n = A.rows();
  dpotrf_(&lower, &n, A.data(), &n, &info);
  if(info != 0) {
    throw std::runtime_error(
        std::string("c++ error: Cholesky decomp failed (for ")
        + std::to_string(n)
        + " x "
        + std::to_string(n)
        + " eigen matrix)");
  }
}

void chol_solve(double* A, int n, double* B, int nrhs) {
  int info;
  dpotrs_(&lower, &n, &nrhs, A, &n, B, &n, &info);
  if(info != 0){ throw std::runtime_error("c++ error: Cholesky solve failed");}
}

void chol_solve(Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  if (A.rows() != B.rows()) {throw std::runtime_error("A.rows() must equal B.rows()");}
  int info;
  int n    = A.rows();
  int nrhs = B.cols();
  dpotrs_(&lower, &n, &nrhs, A.data(), &n, B.data(), &n, &info);
  if(info != 0){ throw std::runtime_error("c++ error: Cholesky solve failed (for eigen matrix)");}
}

/** solves A * X' = B' for X in place */
void chol_solve_t(Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  if (A.rows() != B.cols()) {throw std::runtime_error("A.rows() must equal B.cols()");}
  B.transposeInPlace();
  chol_solve(A, B);
  B.transposeInPlace();
}

