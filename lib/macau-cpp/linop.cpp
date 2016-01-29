#include <Eigen/Dense>
#include <math.h>
#include <cblas.h>
#include <sparse.h>
#include <iostream>

#include <stdexcept>

#include "chol.h"
#include "linop.h"

using namespace Eigen;

void At_mul_A(BlockedSBM & sbm, Eigen::MatrixXd & out) {
  throw std::runtime_error("At_mul_A not implemented.");
}

// out = sbm * b (for vectors)
void A_mul_B(Eigen::VectorXd & out, BlockedSBM & sbm, Eigen::VectorXd & b) {
  if (sbm.nrow != out.size()) {throw std::runtime_error("sbm.nrow must equal out.size()");}
  if (sbm.ncol != b.size())   {throw std::runtime_error("sbm.ncol must equal b.size()");}
  bsbm_A_mul_B( out.data(), & sbm, b.data() );
}

// out' = sbm * B' (for matrices)
void A_mul_Bt(Eigen::MatrixXd & out, BlockedSBM & sbm, Eigen::MatrixXd & B) {
  if (sbm.nrow != out.cols()) {throw std::runtime_error("sbm.nrow must equal out.cols()");}
  if (sbm.ncol != B.cols())   {throw std::runtime_error("sbm.ncol must equal b.cols()");}
  if (out.rows() != B.rows()) {throw std::runtime_error("out.rows() must equal B.rows()");}
  bsbm_A_mul_Bn( out.data(), & sbm, B.data(), B.rows() );
}

void At_mul_A_blas(const Eigen::MatrixXd & A, double* AtA) {
  cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, A.cols(), A.rows(), 1.0, A.data(), A.rows(), 0.0, AtA, A.cols());
}

void A_mul_At_blas(const Eigen::MatrixXd & A, double* AAt) {
  cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, A.rows(), A.cols(), 1.0, A.data(), A.rows(), 0.0, AAt, A.rows());
}

void A_mul_B_blas(Eigen::MatrixXd & Y, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B) {
  if (Y.rows() != A.rows()) {throw std::runtime_error("A.rows() must equal Y.rows()");}
  if (Y.cols() != B.cols()) {throw std::runtime_error("B.cols() must equal Y.cols()");}
  if (A.cols() != B.rows()) {throw std::runtime_error("B.rows() must equal A.cols()");}
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, A.rows(), B.cols(), A.cols(), 1, A.data(), A.rows(), B.data(), B.rows(), 0, Y.data(), Y.rows());
}

void At_mul_B_blas(double beta, Eigen::MatrixXd & Y, double alpha, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B) {
  if (Y.rows() != A.rows()) {throw std::runtime_error("A.rows() must equal Y.rows()");}
  if (Y.cols() != B.cols()) {throw std::runtime_error("B.cols() must equal Y.cols()");}
  if (A.cols() != B.rows()) {throw std::runtime_error("B.rows() must equal A.cols()");}
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, A.rows(), B.cols(), A.cols(), alpha, A.data(), A.rows(), B.data(), B.rows(), beta, Y.data(), Y.rows());
}

void A_mul_Bt_blas(Eigen::MatrixXd & Y, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B) {
  if (Y.rows() != A.rows()) {throw std::runtime_error("A.rows() must equal Y.rows()");}
  if (Y.cols() != B.rows()) {throw std::runtime_error("B.rows() must equal Y.cols()");}
  if (A.cols() != B.cols()) {throw std::runtime_error("B.cols() must equal A.cols()");}
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, A.rows(), B.rows(), A.cols(), 1.0, A.data(), A.rows(), B.data(), B.rows(), 0.0, Y.data(), Y.rows());
}

void Asym_mul_B_left(double beta, Eigen::MatrixXd & Y, double alpha, Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, Y.rows(), Y.cols(), alpha, A.data(), A.rows(), B.data(), B.rows(), beta, Y.data(), Y.rows());
}

void Asym_mul_B_right(double beta, Eigen::MatrixXd & Y, double alpha, Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  cblas_dsymm(CblasColMajor, CblasRight, CblasLower, Y.rows(), Y.cols(), alpha, A.data(), A.rows(), B.data(), B.rows(), beta, Y.data(), Y.rows());
}


// B is in transformed format: [nrhs x nfeat]
int solve(Eigen::MatrixXd & X, SparseFeat & K, double reg, Eigen::MatrixXd & B, double tol) {
  // initialize
  const int nrhs  = B.rows();
  const int nfeat = B.cols();
  double tolsq = tol*tol;

  if (nfeat != K.nfeat()) {throw std::runtime_error("B.cols() must equal K.nfeat()");}

  VectorXd norms(nrhs), inorms(nrhs); 
  norms.setZero();
  inorms.setZero();
#pragma omp parallel for schedule(static)
  for (int rhs = 0; rhs < nrhs; rhs++) {
    double sumsq = 0.0;
    for (int feat = 0; feat < nfeat; feat++) {
      sumsq += B(rhs, feat) * B(rhs, feat);
    }
    norms(rhs)  = sqrt(sumsq);
    inorms(rhs) = 1.0 / norms(rhs);
  }
  MatrixXd R(nrhs, nfeat);
  MatrixXd P(nrhs, nfeat);
  MatrixXd Ptmp(nrhs, nfeat);
  X.setZero();
  // normalize R and P:
#pragma omp parallel for schedule(static) collapse(2)
  for (int feat = 0; feat < nfeat; feat++) {
    for (int rhs = 0; rhs < nrhs; rhs++) {
      R(rhs, feat) = B(rhs, feat) * inorms(rhs);
      P(rhs, feat) = R(rhs, feat);
    }
  }
  MatrixXd* RtR = new MatrixXd(nrhs, nrhs);
  MatrixXd* RtR2 = new MatrixXd(nrhs, nrhs);

  //RtR->setZero();
  //RtR2->setZero();
  MatrixXd KP(nrhs, nfeat);
  MatrixXd KPtmp(nrhs, K.nsamples());
  MatrixXd A(nrhs, nrhs);
  MatrixXd PtKP(nrhs, nrhs);
  MatrixXd Psi(nrhs, nrhs);

  A_mul_At_blas(R, RtR->data());
  // copying A lower tri to upper tri
  for (int i = 1; i < nrhs; i++) {
    for (int j = 0; j < i; j++) {
      (*RtR)(j, i) = (*RtR)(i, j);
    }
  }

  // CG iteration:
  int iter = 0;
  for (iter = 0; iter < 100000; iter++) {
    // solution update:
    A_mul_Bt(KPtmp, K.M, P);
    // move KP += reg * P here with nowait from previous loop
    // http://stackoverflow.com/questions/30496365/parallelize-the-addition-of-a-vector-of-matrices-in-openmp
    A_mul_Bt(KP, K.Mt, KPtmp);
    KP.noalias() += reg * P; // TODO: check if += is parallelized by eigen

    A_mul_Bt_blas(PtKP, P, KP); // TODO: use KPtmp with dsyrk two save 2x time
    chol_decomp(PtKP);
    A = *RtR;
    chol_solve(PtKP, A);
    
    // X += A' * P (as X and P are already transposed)
    At_mul_B_blas(1.0, X, 1.0, A, P);

    // R -= A' * KP (as R and KP are already transposed)
    At_mul_B_blas(1.0, R, -1.0, A, KP);

    // convergence check:
    A_mul_At_blas(R, RtR2->data());
    // copying A lower tri to upper tri
    for (int i = 1; i < nrhs; i++) {
      for (int j = 0; j < i; j++) {
        (*RtR2)(j, i) = (*RtR2)(i, j);
      }
    }

    VectorXd d = RtR2->diagonal();
    //std::cout << "[ iter " << iter << "] " << d.cwiseSqrt() << "\n";
    if ( (d.array() < tolsq).all()) {
      break;
    }
    // Psi = (R R') \ R2 R2'
    chol_decomp(*RtR);
    Psi = *RtR2;
    chol_solve(*RtR, Psi);

    // P = R + Psi' * P (P and R are already transposed)
    At_mul_B_blas(0.0, Ptmp, 1.0, Psi, P);
    P.noalias() = R + Ptmp;
    // R R' = R2 R2'
    std::swap(RtR, RtR2);
  }
  delete RtR;
  delete RtR2;
  return iter;
}
