#ifndef SBM2_H
#define SBM2_H

#include <Eigen/Dense>

extern "C" {
  #include <sparse.h>
}

class SBM2 {
  public:
    BlockedSBM M;
    BlockedSBM Mt;
};

class LinOp {
};

void At_mul_A( Eigen::MatrixXd & out, SBM2 & sbm );
void A_mul_Bt( Eigen::MatrixXd & out, SBM2 & sbm, Eigen::MatrixXd & B );
void At_mul_Bt(Eigen::MatrixXd & out, SBM2 & sbm, Eigen::MatrixXd & B );
void solve(    Eigen::MatrixXd & out, LinOp & linop, Eigen::MatrixXd & B, double tol);

#endif /* SBM2_H */
