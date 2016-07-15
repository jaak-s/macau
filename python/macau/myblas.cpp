#include <stdio.h>

extern "C" void dgemm_(char *transa, char *transb, int *m, int *n, int *k, double *alpha,
           double a[], int *lda, double b[], int *ldb, double *beta, double c[],
           int *ldc);
extern "C" void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
extern "C" void dpotrs_(char *uplo, int* n, int* nrhs, double* A, int* lda, double* B, int* ldb, int* info);

void blasx() {
  int n = 3;
  int nn = n*n;
  double *A = new double[nn];
  double *C = new double[nn];
  A[0] = 0.5; A[1] = 1.4; A[2] = -0.1;
  A[3] = 2.3; A[4] = -.4; A[5] = 19.1;
  A[6] = -.72; A[7] = 0.6; A[8] = 12.3;

  char transA = 'N';
  char transB = 'T';
  double alpha = 1.0;
  double beta  = 0.0;
  dgemm_(&transA, &transB, &n, &n, &n, &alpha, A, &n, A, &n, &beta, C, &n);
}

void lapackx() {
  int n = 3;
  int nn = n*n;
  int nrhs = 2;
  int info;
  char lower = 'L';
  double *A = new double[nn];
  double *B = new double[nrhs*nn];
  A[0] = 6.1;   A[1] = -0.65; A[2] = 5.1;
  A[3] = -0.65; A[4] = 2.4;   A[5] = -0.4;
  A[6] = 5.1;   A[7] = -0.4;  A[8] = 12.3;
  B[0] = 5.2; B[1] = -0.4;
  B[2] = 1.0; B[3] = 1.3;
  B[4] = 0.2; B[5] = -0.15;

  dpotrf_(&lower, &n, A, &n, &info);
  if(info != 0){ printf("c++ error: Cholesky decomp failed"); }
  dpotrs_(&lower, &n, &nrhs, A, &n, B, &n, &info);
  if(info != 0){ printf("c++ error: Cholesky solve failed"); }
}
