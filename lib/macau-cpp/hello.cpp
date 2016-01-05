#include <Eigen/Dense>

using Eigen::MatrixXd;

double hello(void) {
  MatrixXd m(2,2);
  return m(0,0) + 2000;
}
