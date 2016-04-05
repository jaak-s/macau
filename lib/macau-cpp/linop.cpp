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

// out = bcsr * b (for vectors)
void A_mul_B(Eigen::VectorXd & out, CSR & csr, Eigen::VectorXd & b) {
  if (csr.nrow != out.size()) {throw std::runtime_error("csr.nrow must equal out.size()");}
  if (csr.ncol != b.size())   {throw std::runtime_error("csr.ncol must equal b.size()");}
  csr_A_mul_B( out.data(), & csr, b.data() );
}

// OUT' = bcsr * B' (for matrices)
void A_mul_Bt(Eigen::MatrixXd & out, CSR & csr, Eigen::MatrixXd & B) {
  if (csr.nrow != out.cols()) {throw std::runtime_error("csr.nrow must equal out.cols()");}
  if (csr.ncol != B.cols())   {throw std::runtime_error("csr.ncol must equal b.cols()");}
  if (out.rows() != B.rows()) {throw std::runtime_error("out.rows() must equal B.rows()");}
  csr_A_mul_Bn( out.data(), & csr, B.data(), B.rows() );
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

  int ncol = out.cols(), nrow = out.rows();
#pragma omp parallel for schedule(static)
  for (int col = 0; col < ncol; col++) {
    for (int row = 0; row < nrow; row++) {
      out(row, col) += reg * B(row, col);
    }
  }
}

template<>
void AtA_mul_B(Eigen::MatrixXd & out, SparseDoubleFeat & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd tmp) {
  // solution update:
  A_mul_Bt(tmp, A.M, B);
  // move KP += reg * P here with nowait from previous loop
  // http://stackoverflow.com/questions/30496365/parallelize-the-addition-of-a-vector-of-matrices-in-openmp
  A_mul_Bt(out, A.Mt, tmp);
  int ncol = out.cols(), nrow = out.rows();
#pragma omp parallel for schedule(static)
  for (int col = 0; col < ncol; col++) {
    for (int row = 0; row < nrow; row++) {
      out(row, col) += reg * B(row, col);
    }
  }
}

template<>
void AtA_mul_B(Eigen::MatrixXd & out, Eigen::MatrixXd & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd tmp) {
  // solution update:
  A_mul_Bt_blas(tmp, B, A);
  // move KP += reg * P here with nowait from previous loop
  // http://stackoverflow.com/questions/30496365/parallelize-the-addition-of-a-vector-of-matrices-in-openmp
  A_mul_B_blas(out, B, A);

  int ncol = out.cols(), nrow = out.rows();
#pragma omp parallel for schedule(static)
  for (int col = 0; col < ncol; col++) {
    for (int row = 0; row < nrow; row++) {
      out(row, col) += reg * B(row, col);
    }
  }
}

void makeSymmetric(Eigen::MatrixXd & A) {
  A = A.selfadjointView<Eigen::Lower>();
}

/**
 * A is [n x k] matrix
 * returns [n x n] matrix A * A'
 */
Eigen::MatrixXd A_mul_At_combo(Eigen::MatrixXd & A) {
  MatrixXd AAt(A.rows(), A.rows());
  A_mul_At_combo(AAt, A);
  AAt = AAt.selfadjointView<Eigen::Lower>();
  return AAt;
}

/** A   is [n x k] matrix
 *  out is [n x n] matrix
 *  A and out are column-ordered
 *  computes out = A * A'
 *  (storing only lower triangular part)
 */
void A_mul_At_combo(Eigen::MatrixXd & out, Eigen::MatrixXd & A) {
  if (out.rows() >= 128) {
    // using blas for larger matrices
    A_mul_At_blas(A, out.data());
  } else {
    A_mul_At_omp(out, A);
  }
}

void A_mul_At_omp(Eigen::MatrixXd & out, Eigen::MatrixXd & A) {
  const int n = A.rows();
  const int k = A.cols();
  int nthreads = -1;
  double* x = A.data();
  if (A.rows() != out.rows()) {
    throw std::runtime_error("A.rows() must equal out.rows()");
  }

#pragma omp parallel
  {
#pragma omp single
    {
      nthreads = omp_get_num_threads();
    }
  }
  std::vector<MatrixXd> Ys;
  Ys.resize(nthreads, MatrixXd(n, n));

#pragma omp parallel
  {
    const int ithread  = omp_get_thread_num();
    int rows_per_thread = (int) 8 * ceil(k / 8.0 / nthreads);
    int row_start = rows_per_thread * ithread;
    int row_end   = rows_per_thread * (ithread + 1);
    if (row_start >= k) {
      Ys[ithread].setZero();
    } else {
      if (row_end > k) {
        row_end = k;
      }
      double* xi = & x[ row_start * n ];
      int nrows  = row_end - row_start;
      MatrixXd X = Map<MatrixXd>(xi, n, nrows);
      MatrixXd & Y = Ys[ithread];
      Y.triangularView<Eigen::Lower>() = X * X.transpose();
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      double tmp = 0;
      for (int k = 0; k < nthreads; k++) {
        tmp += Ys[k](j, i);
      }
      out(j, i) = tmp;
    }
  }
}

