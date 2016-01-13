#include <Eigen/Dense>
#include <stdio.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

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
