#include "sbm2.h"
#include <Eigen/Dense>
#include <sparse.h>

#include <stdexcept>

void At_mul_A(SBM2 & sbm, Eigen::MatrixXd & out) {
  throw std::runtime_error("At_mul_A not implemented.");
}

// out = sbm * b for vectors 
void A_mul_Bt(Eigen::VectorXd & out, SBM2 & sbm, Eigen::VectorXd & B) {
  // assumes the sizes are right
  bsbm_A_mul_B( out.data(), & sbm.M, B.data() );

}

// out = sbm * B' for matrices
void A_mul_Bt(Eigen::MatrixXd & out, SBM2 & sbm, Eigen::MatrixXd & B) {
  // assumes the sizes are right
  bsbm_A_mul_Bn( out.data(), & sbm.M, B.data(), B.rows() );
}

void At_mul_Bt(Eigen::MatrixXd & out, SBM2 & sbm, Eigen::MatrixXd & B) {
  // assumes the sizes are right
  bsbm_A_mul_Bn( out.data(), & sbm.Mt, B.data(), B.rows() );
}

void solve(Eigen::MatrixXd & out, LinOp & linop, Eigen::MatrixXd & B, double tol) {
  // TODO: cg should be here
}
