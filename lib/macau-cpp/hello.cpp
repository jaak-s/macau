#include <Eigen/Dense>
#include <stdio.h>
#include <cblas.h>
#include "mvnormal.h"
extern "C" {
#include <sparse.h>
}

using namespace Eigen;

void hello(double* x, double* y, int n, int k) {
  //cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, k, 1, x, k, x, k, 0, y, n);
  cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, n, k, 1.0, x, k, 0.0, y, n);
}

void eigenQR(double* x, int nrow, int ncol) {
  MatrixXd X = Map<MatrixXd>(x, nrow, ncol);
  HouseholderQR<MatrixXd> qr(X);
  MatrixXd Q = qr.householderQ();
  printf("Q(0,0) = %f\n", Q(0,0));
}

MatrixXd getx() {
  init_bmrng(100099102);
  MatrixXd x(1000,10);
  bmrandn(x);
  return x;
}

void At_mul_A(const Eigen::MatrixXd & A, Eigen::MatrixXd & C) {
  const int n = A.cols();
  if (n != C.rows()) { printf("A.cols() must equal C.rows()."); exit(1); }
  if (C.rows() != C.cols()) { printf("C.rows() must equal C.cols()."); exit(1); }
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, n, 1, A.data(), n, A.data(), n, 0, C.data(), n);
}
