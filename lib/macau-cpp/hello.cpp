#include <Eigen/Dense>

using Eigen::MatrixXd;

double hello(double* x, int nrows, int ncols) {
  MatrixXd m = Eigen::Map<MatrixXd>(x, nrows, ncols);
  MatrixXd a = m * m.transpose();
  return a(0, 0);
}
