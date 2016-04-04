#ifndef LINOP_H
#define LINOP_H

#include <Eigen/Dense>
#include "chol.h"

extern "C" {
  #include <csr.h>
}

class SparseFeat {
  public:
    BinaryCSR M;
    BinaryCSR Mt;

    SparseFeat() {}

    SparseFeat(int nrow, int ncol, long nnz, int* rows, int* cols) {
      new_bcsr(&M,  nnz, nrow, ncol, rows, cols);
      new_bcsr(&Mt, nnz, ncol, nrow, cols, rows);
    }
    virtual ~SparseFeat() {
      free_bcsr( & M);
      free_bcsr( & Mt);
    }
    int nfeat()    {return M.ncol;}
    int cols()     {return M.ncol;}
    int nsamples() {return M.nrow;}
    int rows()     {return M.nrow;}
};

class SparseDoubleFeat {
  public:
    CSR M;
    CSR Mt;

    SparseDoubleFeat() {}

    SparseDoubleFeat(int nrow, int ncol, long nnz, int* rows, int* cols, double* vals) {
      new_csr(&M,  nnz, nrow, ncol, rows, cols, vals);
      new_csr(&Mt, nnz, ncol, nrow, cols, rows, vals);
    }
    virtual ~SparseDoubleFeat() {
      free_csr( & M);
      free_csr( & Mt);
    }
    int nfeat()    {return M.ncol;}
    int cols()     {return M.ncol;}
    int nsamples() {return M.nrow;}
    int rows()     {return M.nrow;}
};

template<typename T>
void  solve_blockcg(Eigen::MatrixXd & X, T & t, double reg, Eigen::MatrixXd & B, double tol, const int blocksize, const int excess);
template<typename T>
int  solve_blockcg(Eigen::MatrixXd & X, T & t, double reg, Eigen::MatrixXd & B, double tol);
template<typename T>
void At_mul_A(Eigen::MatrixXd & out, T & A);
template<typename T>
void compute_uhat(Eigen::MatrixXd & uhat, T & feat, Eigen::MatrixXd & beta);
template<typename T>
void AtA_mul_B(Eigen::MatrixXd & out, T & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd tmp);


void At_mul_B_blas(double beta, Eigen::MatrixXd & Y, double alpha, Eigen::MatrixXd & A, Eigen::MatrixXd & B);
void At_mul_A_blas(const Eigen::MatrixXd & A, double* AtA);
void A_mul_At_blas(const Eigen::MatrixXd & A, double* AAt);
void A_mul_B_blas(Eigen::MatrixXd & Y, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B);
void A_mul_Bt_blas(Eigen::MatrixXd & Y, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B);
//void compute_uhat( Eigen::MatrixXd & uhat, Eigen::MatrixXd & denseFeat, Eigen::MatrixXd & beta);
void A_mul_At_combo(Eigen::MatrixXd & out, Eigen::MatrixXd & A);
void A_mul_At_omp(Eigen::MatrixXd & out, Eigen::MatrixXd & A);
Eigen::MatrixXd A_mul_At_combo(Eigen::MatrixXd & A);

// util functions:
void A_mul_B(  Eigen::VectorXd & out, BinaryCSR & csr, Eigen::VectorXd & b);
void A_mul_Bt( Eigen::MatrixXd & out, BinaryCSR & csr, Eigen::MatrixXd & B);
void A_mul_B(  Eigen::VectorXd & out, CSR & csr, Eigen::VectorXd & b);
void A_mul_Bt( Eigen::MatrixXd & out, CSR & csr, Eigen::MatrixXd & B);

void makeSymmetric(Eigen::MatrixXd & A);

// Y = beta * Y + alpha * A * B (where B is symmetric)
void Asym_mul_B_left(double beta, Eigen::MatrixXd & Y, double alpha, Eigen::MatrixXd & A, Eigen::MatrixXd & B);
void Asym_mul_B_right(double beta, Eigen::MatrixXd & Y, double alpha, Eigen::MatrixXd & A, Eigen::MatrixXd & B);

///////////////////////////////////
//     Template functions
///////////////////////////////////

//// for Sparse
/** 
 * uhat       - [D x N] dense matrix
 * sparseFeat - [N x F] sparse matrix (features)
 * beta       - [D x F] dense matrix
 * computes:
 *   uhat = beta * sparseFeat'
 */
template<> inline void compute_uhat(Eigen::MatrixXd & uhat, SparseFeat & sparseFeat, Eigen::MatrixXd & beta) {
  A_mul_Bt(uhat, sparseFeat.M, beta);
}

template<> inline void compute_uhat(Eigen::MatrixXd & uhat, SparseDoubleFeat & feat, Eigen::MatrixXd & beta) {
  A_mul_Bt(uhat, feat.M, beta);
}

/** computes uhat = denseFeat * beta, where beta and uhat are row ordered */
template<> inline void compute_uhat(Eigen::MatrixXd & uhat, Eigen::MatrixXd & denseFeat, Eigen::MatrixXd & beta) {
  A_mul_Bt_blas(uhat, beta, denseFeat);
}

template<>
inline void At_mul_A(Eigen::MatrixXd & out, SparseFeat & A) {
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

template<>
inline void At_mul_A(Eigen::MatrixXd & out, SparseDoubleFeat & A) {
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

template <>
inline void At_mul_A(Eigen::MatrixXd & out, Eigen::MatrixXd & A) {
  // TODO: use blas
  out.triangularView<Eigen::Lower>() = A.transpose() * A;
}

/** good values for solve_blockcg are blocksize=32 an excess=8 */
template<typename T>
inline void solve_blockcg(Eigen::MatrixXd & X, T & K, double reg, Eigen::MatrixXd & B, double tol, const int blocksize, const int excess) {
  if (B.rows() <= excess + blocksize) {
    solve_blockcg(X, K, reg, B, tol);
    return;
  }
  // split B into blocks of size <blocksize> (+ excess if needed)
  Eigen::MatrixXd Xblock, Bblock;
  for (int i = 0; i < B.rows(); i += blocksize) {
    int nrows = blocksize;
    if (i + blocksize + excess >= B.rows()) {
      nrows = B.rows() - i;
    }
    Bblock.resize(nrows, B.cols());
    Xblock.resize(nrows, X.cols());

    Bblock = B.block(i, 0, nrows, B.cols());
    solve_blockcg(Xblock, K, reg, Bblock, tol);
    X.block(i, 0, nrows, X.cols()) = Xblock;
  }
}

template<typename T>
inline int solve_blockcg(Eigen::MatrixXd & X, T & K, double reg, Eigen::MatrixXd & B, double tol) {
  // initialize
  const int nrhs  = B.rows();
  const int nfeat = B.cols();
  double tolsq = tol*tol;

  if (nfeat != K.cols()) {throw std::runtime_error("B.cols() must equal K.cols()");}

  Eigen::VectorXd norms(nrhs), inorms(nrhs); 
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
  Eigen::MatrixXd R(nrhs, nfeat);
  Eigen::MatrixXd P(nrhs, nfeat);
  Eigen::MatrixXd Ptmp(nrhs, nfeat);
  X.setZero();
  // normalize R and P:
#pragma omp parallel for schedule(static) collapse(2)
  for (int feat = 0; feat < nfeat; feat++) {
    for (int rhs = 0; rhs < nrhs; rhs++) {
      R(rhs, feat) = B(rhs, feat) * inorms(rhs);
      P(rhs, feat) = R(rhs, feat);
    }
  }
  Eigen::MatrixXd* RtR = new Eigen::MatrixXd(nrhs, nrhs);
  Eigen::MatrixXd* RtR2 = new Eigen::MatrixXd(nrhs, nrhs);

  //RtR->setZero();
  //RtR2->setZero();
  Eigen::MatrixXd KP(nrhs, nfeat);
  Eigen::MatrixXd KPtmp(nrhs, K.rows());
  Eigen::MatrixXd A(nrhs, nrhs);
  Eigen::MatrixXd PtKP(nrhs, nrhs);
  Eigen::MatrixXd Psi(nrhs, nrhs);

  A_mul_At_blas(R, RtR->data());
  makeSymmetric(*RtR);

  // CG iteration:
  int iter = 0;
  for (iter = 0; iter < 100000; iter++) {
    // KP = K * P
    AtA_mul_B(KP, K, reg, P, KPtmp);

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

    Eigen::VectorXd d = RtR2->diagonal();
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
  // unnormalizing X:
#pragma omp parallel for schedule(static) collapse(2)
  for (int feat = 0; feat < nfeat; feat++) {
    for (int rhs = 0; rhs < nrhs; rhs++) {
      X(rhs, feat) *= norms(rhs);
    }
  }
  delete RtR;
  delete RtR2;
  return iter;
}

#endif /* LINOP_H */
