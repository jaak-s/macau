#include <Eigen/Dense>
#include <math.h>
extern "C" {
  #include <cblas.h>
  #include <csr.h>
}
#include <iostream>

#include <stdexcept>

#include "chol.h"
#include "linop.h"

using namespace Eigen;
using namespace std;

// out = bcsr * b (for vectors)
void A_mul_B(Eigen::VectorXd & out, BinaryCSR & csr, Eigen::VectorXd & b) {
  if (csr.nrow != out.size()) {throw std::runtime_error("csr.nrow must equal out.size()");}
  if (csr.ncol != b.size())   {throw std::runtime_error("csr.ncol must equal b.size()");}
  bcsr_A_mul_B( out.data(), & csr, b.data() );
}

// OUT' = bcsr * B' (for matrices)
void A_mul_Bt(Eigen::MatrixXd & out, BinaryCSR & csr, Eigen::MatrixXd & B) {
  if (csr.nrow != out.cols()) {throw std::runtime_error("csr.nrow must equal out.cols()");}
  if (csr.ncol != B.cols())   {throw std::runtime_error("csr.ncol must equal b.cols()");}
  if (out.rows() != B.rows()) {throw std::runtime_error("out.rows() must equal B.rows()");}
  bcsr_A_mul_Bn( out.data(), & csr, B.data(), B.rows() );
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

void At_mul_B_blas(double beta, Eigen::MatrixXd & Y, double alpha, Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
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

template<> void AtA_mul_B(Eigen::MatrixXd & out, SparseFeat & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd tmp) {
  // solution update:
  A_mul_Bt(tmp, A.M, B);
  // move KP += reg * P here with nowait from previous loop
  // http://stackoverflow.com/questions/30496365/parallelize-the-addition-of-a-vector-of-matrices-in-openmp
  A_mul_Bt(out, A.Mt, tmp);
  out.noalias() += reg * B; // TODO: check if += is parallelized by eigen
}

template<> void AtA_mul_B(Eigen::MatrixXd & out, Eigen::MatrixXd & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd tmp) {
  // solution update:
  A_mul_Bt_blas(tmp, B, A);
  // move KP += reg * P here with nowait from previous loop
  // http://stackoverflow.com/questions/30496365/parallelize-the-addition-of-a-vector-of-matrices-in-openmp
  A_mul_B_blas(out, B, A);
  out.noalias() += reg * B; // TODO: check if += is parallelized by eigen
}

// B is in transformed format: [nrhs x nfeat]
