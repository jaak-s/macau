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

MatrixXd getx() {
  MatrixXd x(3,2);
  x.setZero();
  x(0,0) = 1;
  x(1,0) = 2;
  x(2,0) = 3;
  return x;
}
