#include <Eigen/Dense>
#include <stdio.h>

#include <Python.h>
#include "macau_api.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

double hellotest() {
  import_macau();
  VectorXd x(10);
  x.setZero();
  x[0] = 18;
  return test1(&x);
}

double hello(double* x, int nrows, int ncols) {
  MatrixXd m = Eigen::Map<MatrixXd>(x, nrows, ncols);
  MatrixXd a = m * m.transpose();
  return a(0, 0);
}

void solve(double* Araw, double* braw, double* out, int n) {
  MatrixXd A = Eigen::Map<MatrixXd>(Araw, n, n);
  VectorXd b = Eigen::Map<VectorXd>(braw, n);
  VectorXd k = A.ldlt().solve(b);
  for (int i = 0; i < n; i++) {
    out[i] = k(i);
  }
}
