#include <Eigen/Dense>
#include <math.h>
extern "C" {
  #include <csr.h>
}
#include <iostream>

#include <stdexcept>

#include "chol.h"
#include "linop.h"

using namespace Eigen;
using namespace std;

extern "C" void dsyrk_(char *uplo, char *trans, int *m, int *n, double *alpha, double a[],
            int *lda, double *beta, double c[], int *ldc);
extern "C" void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha,
            double a[], int *lda, double b[], int *ldb, double *beta, double c[],
            int *ldc);
/*
extern "C" void dsymm_(char *side, char *uplo, int *m, int *n, double *alpha, double a[],
            int *lda, double b[], int *ldb, double *beta, double c[], int *ldc);
*/

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

void At_mul_A(Eigen::MatrixXd & out, SparseFeat & A) {
  if (out.cols() != A.cols()) {
    throw std::runtime_error("At_mul_A(SparseFeat): out.cols() must equal A.cols()");
  }
  if (out.cols() != out.rows()) {
    throw std::runtime_error("At_mul_A(SparseFeat): out must be square matrix.)");
  }

  out.setZero();
  const int nfeat = A.M.ncol;

#pragma omp parallel for schedule(dynamic, 8)
  for (int f1 = 0; f1 < nfeat; f1++) {
    int end = A.Mt.row_ptr[f1 + 1];
    out(f1, f1) = end - A.Mt.row_ptr[f1];
    // looping over all non-zero rows of f1
    for (int i = A.Mt.row_ptr[f1]; i < end; i++) {
      int Mrow = A.Mt.cols[i]; /* row in M */
      int end2 = A.M.row_ptr[Mrow + 1];
      for (int j = A.M.row_ptr[Mrow]; j < end2; j++) {
        int f2 = A.M.cols[j];
        if (f1 < f2) {
          out(f2, f1) += 1;
        }
      }
    }
  }
}

void At_mul_A(Eigen::MatrixXd & out, Eigen::MatrixXd & A) {
  // TODO: use blas
  out.triangularView<Eigen::Lower>() = A.transpose() * A;
}

void At_mul_A(Eigen::MatrixXd & out, SparseDoubleFeat & A) {
  if (out.cols() != A.cols()) {
    throw std::runtime_error("At_mul_A(SparseDoubleFeat): out.cols() must equal A.cols()");
  }
  if (out.cols() != out.rows()) {
    throw std::runtime_error("At_mul_A(SparseDoubleFeat): out must be square matrix.)");
  }
  out.setZero();
  const int nfeat = A.M.ncol;

#pragma omp parallel for schedule(dynamic, 8)
  for (int f1 = 0; f1 < nfeat; f1++) {
    // looping over all non-zero rows of f1
    for (int i = A.Mt.row_ptr[f1], end = A.Mt.row_ptr[f1 + 1]; i < end; i++) {
      int Mrow     = A.Mt.cols[i]; /* row in M */
      double val1  = A.Mt.vals[i]; /* value for Mrow */

      for (int j = A.M.row_ptr[Mrow], end2 = A.M.row_ptr[Mrow + 1]; j < end2; j++) {
        int f2 = A.M.cols[j];
        if (f1 <= f2) {
          out(f2, f1) += A.M.vals[j] * val1;
        }
      }
    }
  }
}

void At_mul_A_blas(Eigen::MatrixXd & A, double* AtA) {
  char lower  = 'L';
  char trans  = 'T';
  int  m      = A.cols();
  int  n      = A.rows();
  double one  = 1.0;
  double zero = 0.0;
  dsyrk_(&lower, &trans, &m, &n, &one, A.data(),
         &n, &zero, AtA, &m);
}

void A_mul_At_blas(Eigen::MatrixXd & A, double* AAt) {
  char lower  = 'L';
  char trans  = 'N';
  int  m      = A.cols();
  int  n      = A.rows();
  double one  = 1.0;
  double zero = 0.0;
  dsyrk_(&lower, &trans, &n, &m, &one, A.data(),
         &n, &zero, AAt, &n);
}

void A_mul_B_blas(Eigen::MatrixXd & Y, Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  if (Y.rows() != A.rows()) {throw std::runtime_error("A.rows() must equal Y.rows()");}
  if (Y.cols() != B.cols()) {throw std::runtime_error("B.cols() must equal Y.cols()");}
  if (A.cols() != B.rows()) {throw std::runtime_error("B.rows() must equal A.cols()");}
  int m = A.rows();
  int n = B.cols();
  int k = A.cols();
  char transA = 'N';
  char transB = 'N';
  double alpha = 1.0;
  double beta  = 0.0;
  dgemm_(&transA, &transB, &m, &n, &k, &alpha, A.data(), &m, B.data(), &k, &beta, Y.data(), &m);
}

void At_mul_B_blas(Eigen::MatrixXd & Y, Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  if (Y.rows() != A.cols()) {throw std::runtime_error("A.rows() must equal Y.rows()");}
  if (Y.cols() != B.cols()) {throw std::runtime_error("B.cols() must equal Y.cols()");}
  if (A.rows() != B.rows()) {throw std::runtime_error("B.rows() must equal A.cols()");}
  int m = A.cols();
  int n = B.cols();
  int k = A.rows();
  char transA = 'T';
  char transB = 'N';
  double alpha = 1.0;
  double beta  = 0.0;
  dgemm_(&transA, &transB, &m, &n, &k, &alpha, A.data(), &k, B.data(), &k, &beta, Y.data(), &m);
}

void A_mul_Bt_blas(Eigen::MatrixXd & Y, Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  if (Y.rows() != A.rows()) {throw std::runtime_error("A.rows() must equal Y.rows()");}
  if (Y.cols() != B.rows()) {throw std::runtime_error("B.rows() must equal Y.cols()");}
  if (A.cols() != B.cols()) {throw std::runtime_error("B.cols() must equal A.cols()");}
  int m = A.rows();
  int n = B.rows();
  int k = A.cols();
  char transA = 'N';
  char transB = 'T';
  double alpha = 1.0;
  double beta  = 0.0;
  dgemm_(&transA, &transB, &m, &n, &k, &alpha, A.data(), &m, B.data(), &n, &beta, Y.data(), &m);
}
/*
void Asym_mul_B_left(double beta, Eigen::MatrixXd & Y, double alpha, Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  cblas_dsymm(CblasColMajor, CblasLeft, CblasLower, Y.rows(), Y.cols(), alpha, A.data(), A.rows(), B.data(), B.rows(), beta, Y.data(), Y.rows());
}

void Asym_mul_B_right(double beta, Eigen::MatrixXd & Y, double alpha, Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  cblas_dsymm(CblasColMajor, CblasRight, CblasLower, Y.rows(), Y.cols(), alpha, A.data(), A.rows(), B.data(), B.rows(), beta, Y.data(), Y.rows());
}*/

template<> void AtA_mul_B(Eigen::MatrixXd & out, SparseFeat & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd & tmp) {
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
void AtA_mul_B(Eigen::MatrixXd & out, SparseDoubleFeat & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd & tmp) {
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
void AtA_mul_B(Eigen::MatrixXd & out, Eigen::MatrixXd & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd & tmp) {
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

void A_mul_Bt_omp_sym(Eigen::MatrixXd & out, Eigen::MatrixXd & A, Eigen::MatrixXd & B) {
  const int n = A.rows();
  const int k = A.cols();
  int nthreads = -1;
  double* x  = A.data();
  double* x2 = B.data();
  assert(A.rows() == B.rows());
  assert(A.cols() == B.cols());
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
      double* xi  = & x[ row_start * n ];
      double* x2i = & x2[ row_start * n ];
      int nrows   = row_end - row_start;
      MatrixXd X  = Map<MatrixXd>(xi, n, nrows);
      MatrixXd X2 = Map<MatrixXd>(x2i, n, nrows);
      MatrixXd & Y = Ys[ithread];
      Y.triangularView<Eigen::Lower>() = X * X2.transpose();
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

Eigen::VectorXd col_square_sum(SparseFeat & A) {
  const int ncol = A.cols();
  VectorXd out(ncol);
#pragma omp parallel for schedule(static)
  for (int col = 0; col < ncol; col++) {
    out(col) = A.Mt.row_ptr[col + 1] - A.Mt.row_ptr[col];
  }
  return out;
}

Eigen::VectorXd col_square_sum(SparseDoubleFeat & A) {
  const int ncol = A.cols();
  const int* row_ptr = A.Mt.row_ptr;
  const double* vals = A.Mt.vals;
  VectorXd out(ncol);

#pragma omp parallel for schedule(dynamic, 256)
  for (int col = 0; col < ncol; col++) {
    double tmp = 0;
    int i   = row_ptr[col];
    int end = row_ptr[col + 1];
    for (; i < end; i++) {
      tmp += vals[i] * vals[i];
    }
    out(col) = tmp;
  }
  return out;
}
