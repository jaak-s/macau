#include <Eigen/Dense>

void hello(double* x, double* y, int n, int k);
void hello2(double* x, double* y, int n, int k);
Eigen::MatrixXd getx();
void At_mul_A_eig(Eigen::MatrixXd & A, Eigen::MatrixXd & C);
void eigenQR(double* x, int nrow, int ncol);
