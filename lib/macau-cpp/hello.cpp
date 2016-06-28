#include <Eigen/Dense>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "mvnormal.h"
extern "C" {
#include <sparse.h>
}

extern "C" void dsyrk_(char *uplo, char *trans, int *m, int *n, double *alpha, double a[],
            int *lda, double *beta, double c[], int *ldc);

using namespace Eigen;

void hello(double* x, double* y, int n, int k) {
  //cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, n, k, 1.0, x, k, 0.0, y, n);
  char lower  = 'L';
  char trans  = 'T';
  double one  = 1.0;
  double zero = 0.0;
  dsyrk_(&lower, &trans, &n, &k, &one, x, &k, &zero, y, &n);
}

void eigenQR(double* x, int nrow, int ncol) {
  MatrixXd X = Map<MatrixXd>(x, nrow, ncol);
  HouseholderQR<MatrixXd> qr(X);
  MatrixXd Q = qr.householderQ();
  printf("Q(0,0) = %f\n", Q(0,0));
}

MatrixXd getx() {
  init_bmrng(100099102);
  MatrixXd x(1000,10);
  bmrandn(x);
  return x;
}

/** x is [n x k] matrix
 *  y is [n x n] matrix
 *  x and y are column-ordered
 *  computes y = x * x'
 *  (storing only lower triangular part)
 */
void hello2(double* x, double* y, int n, int k) {
  if (n >= 256) {
    // probably broken
    //cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans, n, k, 1.0, x, k, 0.0, y, n);
    char lower  = 'L';
    char trans  = 'N';
    double one  = 1.0;
    double zero = 0.0;
    dsyrk_(&lower, &trans, &n, &k, &one, x, &n, &zero, y, &n);
    return;
  }
  int nthreads = -1;
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
    if (row_end > k) {
      row_end = k;
    }
    double* xi = & x[ row_start * n ];
    int nrows  = row_end - row_start;
    MatrixXd X = Map<MatrixXd>(xi, n, nrows);
    MatrixXd & Y = Ys[ithread];
    Y.triangularView<Eigen::Lower>() = X * X.transpose();
  }
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n; j++) {
      double tmp = 0;
      for (int k = 0; k < nthreads; k++) {
        tmp += Ys[k](j, i);
      }
      y[i*n + j] = tmp;
    }
  }
}

void At_mul_A_eig(Eigen::MatrixXd & A, Eigen::MatrixXd & C) {
  const int n = A.cols();
  if (n != C.rows()) { printf("A.cols() must equal C.rows()."); exit(1); }
  if (C.rows() != C.cols()) { printf("C.rows() must equal C.cols()."); exit(1); }
  C.triangularView<Eigen::Lower>() = A.transpose() * A;
}
