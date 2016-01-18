#include <Eigen/Dense>
#include <stdio.h>
#include <cblas.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

double hello(double* x, int nrows, int ncols) {
  MatrixXd m = Eigen::Map<MatrixXd>(x, nrows, ncols);
  MatrixXd a = m * m.transpose();
  return a(0, 0);
}

MatrixXd getx() {
  MatrixXd x(3,2);
  x.setZero();
  x(0,0) = 1;
  x(1,0) = 2;
  x(2,0) = 3;
  return x;
}

void At_mul_A(const Eigen::MatrixXd & A, Eigen::MatrixXd & C) {
  const int n = A.cols();
  if (n != C.rows()) { printf("A.cols() must equal C.rows()."); exit(1); }
  if (C.rows() != C.cols()) { printf("C.rows() must equal C.cols()."); exit(1); }
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, n, n, n, 1, A.data(), n, A.data(), n, 0, C.data(), n);
}
