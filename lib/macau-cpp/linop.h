#ifndef LINOP_H
#define LINOP_H

#include <Eigen/Dense>
#include "chol.h"
#include "bpmfutils.h"

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
template<typename T, unsigned N>
inline int solve_blockcgx(Eigen::MatrixXd & X, T & K, double reg, Eigen::MatrixXd & B, double tol);

template<typename T>
void At_mul_A(Eigen::MatrixXd & out, T & A);
template<typename T>
void compute_uhat(Eigen::MatrixXd & uhat, T & feat, Eigen::MatrixXd & beta);
template<typename T>
void AtA_mul_B(Eigen::MatrixXd & out, T & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd tmp);

// compile-time optimized versions (N - number of RHSs)
template<unsigned N>
void AtA_mul_Bx(Eigen::MatrixXd & out, SparseFeat & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd tmp);
template<unsigned N>
void AtA_mul_Bx(Eigen::MatrixXd & out, SparseDoubleFeat & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd tmp);
template<unsigned N>
void A_mul_Bx(Eigen::MatrixXd & out, BinaryCSR & A, Eigen::MatrixXd & B);
template<unsigned N>
void A_mul_Bx(Eigen::MatrixXd & out, CSR & A, Eigen::MatrixXd & B);


void At_mul_B_blas(double beta, Eigen::MatrixXd & Y, double alpha, Eigen::MatrixXd & A, Eigen::MatrixXd & B);
void At_mul_A_blas(const Eigen::MatrixXd & A, double* AtA);
void A_mul_At_blas(const Eigen::MatrixXd & A, double* AAt);
void A_mul_B_blas(Eigen::MatrixXd & Y, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B);
void A_mul_Bt_blas(Eigen::MatrixXd & Y, const Eigen::MatrixXd & A, const Eigen::MatrixXd & B);

//template<int N>
//inline void A_mul_B_omp(double alpha, Eigen::MatrixXd & out, double beta, Eigen::Matrix<double, N, N> & A, Eigen::MatrixXd & B);
inline void A_mul_B_omp(double alpha, Eigen::MatrixXd & out, double beta, Eigen::MatrixXd & A, Eigen::MatrixXd & B);
/*
template<int N>
inline void A_mul_B_add_omp(Eigen::MatrixXd & out, Eigen::Matrix<double, N, N> & A, Eigen::MatrixXd & B);
template<int N>
inline void A_mul_B_sub_omp(Eigen::MatrixXd & out, Eigen::Matrix<double, N, N> & A, Eigen::MatrixXd & B);
*/

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
  switch(B.rows()) {
    case 1: return solve_blockcgx<T,1>(X, K, reg, B, tol);
    case 2: return solve_blockcgx<T,2>(X, K, reg, B, tol);
    case 3: return solve_blockcgx<T,3>(X, K, reg, B, tol);
    case 4: return solve_blockcgx<T,4>(X, K, reg, B, tol);
    case 5: return solve_blockcgx<T,5>(X, K, reg, B, tol);

    case 6: return solve_blockcgx<T,6>(X, K, reg, B, tol);
    case 7: return solve_blockcgx<T,7>(X, K, reg, B, tol);
    case 8: return solve_blockcgx<T,8>(X, K, reg, B, tol);
    case 9: return solve_blockcgx<T,9>(X, K, reg, B, tol);
    case 10: return solve_blockcgx<T,10>(X, K, reg, B, tol);

    case 11: return solve_blockcgx<T,11>(X, K, reg, B, tol);
    case 12: return solve_blockcgx<T,12>(X, K, reg, B, tol);
    case 13: return solve_blockcgx<T,13>(X, K, reg, B, tol);
    case 14: return solve_blockcgx<T,14>(X, K, reg, B, tol);
    case 15: return solve_blockcgx<T,15>(X, K, reg, B, tol);

    case 16: return solve_blockcgx<T,16>(X, K, reg, B, tol);
    case 17: return solve_blockcgx<T,17>(X, K, reg, B, tol);
    case 18: return solve_blockcgx<T,18>(X, K, reg, B, tol);
    case 19: return solve_blockcgx<T,19>(X, K, reg, B, tol);
    case 20: return solve_blockcgx<T,20>(X, K, reg, B, tol);

    case 21: return solve_blockcgx<T,21>(X, K, reg, B, tol);
    case 22: return solve_blockcgx<T,22>(X, K, reg, B, tol);
    case 23: return solve_blockcgx<T,23>(X, K, reg, B, tol);
    case 24: return solve_blockcgx<T,24>(X, K, reg, B, tol);
    case 25: return solve_blockcgx<T,25>(X, K, reg, B, tol);

    case 26: return solve_blockcgx<T,26>(X, K, reg, B, tol);
    case 27: return solve_blockcgx<T,27>(X, K, reg, B, tol);
    case 28: return solve_blockcgx<T,28>(X, K, reg, B, tol);
    case 29: return solve_blockcgx<T,29>(X, K, reg, B, tol);
    case 30: return solve_blockcgx<T,30>(X, K, reg, B, tol);

    case 31: return solve_blockcgx<T,31>(X, K, reg, B, tol);
    case 32: return solve_blockcgx<T,32>(X, K, reg, B, tol);
    case 33: return solve_blockcgx<T,33>(X, K, reg, B, tol);
    case 34: return solve_blockcgx<T,34>(X, K, reg, B, tol);
    case 35: return solve_blockcgx<T,35>(X, K, reg, B, tol);

    case 36: return solve_blockcgx<T,36>(X, K, reg, B, tol);
    case 37: return solve_blockcgx<T,37>(X, K, reg, B, tol);
    case 38: return solve_blockcgx<T,38>(X, K, reg, B, tol);
    case 39: return solve_blockcgx<T,39>(X, K, reg, B, tol);
    case 40: return solve_blockcgx<T,40>(X, K, reg, B, tol);
    default: throw std::runtime_error("BlockCG only available for up to 40 RHSs.");
  }
}

template<typename T, unsigned N>
inline int solve_blockcgx(Eigen::MatrixXd & X, T & K, double reg, Eigen::MatrixXd & B, double tol) {
  // initialize
  const int nfeat = B.cols();
  const int nrhs  = B.rows();
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

  Eigen::MatrixXd KP(nrhs, nfeat);
  Eigen::MatrixXd KPtmp(nrhs, K.rows());
  Eigen::MatrixXd PtKP(nrhs, nrhs);
  //Eigen::Matrix<double, N, N> A;
  //Eigen::Matrix<double, N, N> Psi;
  Eigen::MatrixXd A;
  Eigen::MatrixXd Psi;


  A_mul_At_blas(R, RtR->data());
  makeSymmetric(*RtR);

  const int nblocks = (int)ceil(nfeat / 64.0);

  // CG iteration:
  int iter = 0;
  for (iter = 0; iter < 100000; iter++) {
    // KP = K * P
    double t1 = tick();
    AtA_mul_Bx<N>(KP, K, reg, P, KPtmp);
    double t2 = tick();

    A_mul_Bt_blas(PtKP, P, KP); // TODO: use KPtmp with dsyrk two save 2x time
    Eigen::LLT<Eigen::MatrixXd> chol = PtKP.llt();
    A = chol.solve(*RtR);
    A.transposeInPlace();
    double t3 = tick();
    
#pragma omp parallel for schedule(dynamic, 4)
    for (int block = 0; block < nblocks; block++) {
      int col = block * 64;
      int bcols = std::min(64, nfeat - col);
      // X += A' * P
      X.block(0, col, nrhs, bcols).noalias() += A *  P.block(0, col, nrhs, bcols);
      // R -= A' * KP
      R.block(0, col, nrhs, bcols).noalias() -= A * KP.block(0, col, nrhs, bcols);
    }
    double t4 = tick();

    // convergence check:
    A_mul_At_combo(*RtR2, R);
    makeSymmetric(*RtR2);

    Eigen::VectorXd d = RtR2->diagonal();
    //std::cout << "[ iter " << iter << "] " << d.cwiseSqrt() << "\n";
    if ( (d.array() < tolsq).all()) {
      break;
    }
    // Psi = (R R') \ R2 R2'
    chol = RtR->llt();
    Psi  = chol.solve(*RtR2);
    Psi.transposeInPlace();
    double t5 = tick();

    // P = R + Psi' * P (P and R are already transposed)
#pragma omp parallel for schedule(dynamic, 8)
    for (int block = 0; block < nblocks; block++) {
      int col = block * 64;
      int bcols = std::min(64, nfeat - col);
      Eigen::MatrixXd xtmp(nrhs, bcols);
      xtmp = Psi *  P.block(0, col, nrhs, bcols);
      P.block(0, col, nrhs, bcols) = R.block(0, col, nrhs, bcols) + xtmp;
    }

    // R R' = R2 R2'
    std::swap(RtR, RtR2);
    double t6 = tick();
    printf("t2-t1 = %.3f, t3-t2 = %.3f, t4-t3 = %.3f, t5-t4 = %.3f, t6-t5 = %.3f\n", t2-t1, t3-t2, t4-t3, t5-t4, t6-t5);
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

template<unsigned N>
void A_mul_Bx(Eigen::MatrixXd & out, BinaryCSR & A, Eigen::MatrixXd & B) {
  assert(N == out.rows());
  assert(N == B.rows());
  assert(A.ncol == B.cols());
  assert(A.nrow == out.cols());

  int* row_ptr   = A.row_ptr;
  int* cols      = A.cols;
  const int nrow = A.nrow;
  double* Y = out.data();
  double* X = B.data();
#pragma omp parallel for schedule(dynamic, 256)
  for (int row = 0; row < nrow; row++) {
    double tmp[N] = { 0 };
    const int end = row_ptr[row + 1];
    for (int i = row_ptr[row]; i < end; i++) {
      int col = cols[i] * N;
      for (int j = 0; j < N; j++) {
         tmp[j] += X[col + j];
      }
    }
    int r = row * N;
    for (int j = 0; j < N; j++) {
      Y[r + j] = tmp[j];
    }
  }
}

template<unsigned N>
void A_mul_Bx(Eigen::MatrixXd & out, CSR & A, Eigen::MatrixXd & B) {
  assert(N == out.rows());
  assert(N == B.rows());
  assert(A.ncol == B.cols());
  assert(A.nrow == out.cols());

  int* row_ptr   = A.row_ptr;
  int* cols      = A.cols;
  double* vals   = A.vals;
  const int nrow = A.nrow;
  double* Y = out.data();
  double* X = B.data();
#pragma omp parallel for schedule(dynamic, 256)
  for (int row = 0; row < nrow; row++) {
    double tmp[N] = { 0 };
    const int end = row_ptr[row + 1];
    for (int i = row_ptr[row]; i < end; i++) {
      int col = cols[i] * N;
      double val = vals[i];
      for (int j = 0; j < N; j++) {
         tmp[j] += X[col + j] * val;
      }
    }
    int r = row * N;
    for (int j = 0; j < N; j++) {
      Y[r + j] = tmp[j];
    }
  }
}

template<unsigned N>
void AtA_mul_Bx(Eigen::MatrixXd & out, SparseFeat & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd tmp) {
  A_mul_Bx<N>(tmp, A.M,  B);
  A_mul_Bx<N>(out, A.Mt, tmp);

  const int ncol = out.cols();
  const int nrow = out.rows();
#pragma omp parallel for schedule(static) collapse(2)
  for (int col = 0; col < ncol; col++) {
    for (int row = 0; row < nrow; row++) {
      out(row, col) += reg * B(row, col);
    }
  }
}

template<unsigned N>
void AtA_mul_Bx(Eigen::MatrixXd & out, SparseDoubleFeat & A, double reg, Eigen::MatrixXd & B, Eigen::MatrixXd tmp) {
  A_mul_Bx<N>(tmp, A.M,  B);
  A_mul_Bx<N>(out, A.Mt, tmp);

  const int ncol = out.cols();
  const int nrow = out.rows();
#pragma omp parallel for schedule(static) collapse(2)
  for (int col = 0; col < ncol; col++) {
    for (int row = 0; row < nrow; row++) {
      out(row, col) += reg * B(row, col);
    }
  }
}

// computes out = alpha * out + beta * A * B
inline void A_mul_B_omp(
    double alpha,
    Eigen::MatrixXd & out,
    double beta,
    Eigen::MatrixXd & A,
    Eigen::MatrixXd & B)
{
  assert(out.cols() == B.cols());
  const int nblocks = (int)ceil(out.cols() / 64.0);
  const int nrow = out.rows();
  const int ncol = out.cols();
#pragma omp parallel for schedule(dynamic, 8)
  for (int block = 0; block < nblocks; block++) {
    int col = block * 64;
    int bcols = std::min(64, ncol - col);
    out.template block(0, col, nrow, bcols).noalias() = alpha * out.template block(0, col, nrow, bcols) + beta * A * B.template block(0, col, nrow, bcols);
  }
}

/*
template<int N>
inline void A_mul_B_omp(
    double alpha,
    Eigen::MatrixXd & out,
    double beta,
    Eigen::Matrix<double, N, N> & A,
    Eigen::MatrixXd & B)
{
  assert(out.cols() == B.cols());
  const int nblocks = out.cols() / 64;
#pragma omp parallel for schedule(dynamic, 8)
  for (int block = 0; block < nblocks; block++) {
    int col = block * 64;
    out.template block<N, 64>(0, col).noalias() = alpha * out.template block<N, 64>(0, col) + beta * A * B.template block<N, 64>(0, col);
  }
  // last remaining block
  int col = nblocks * 64;
  out.block(0, col, N, out.cols() - col) = alpha * out.block(0, col, N, out.cols() - col) + beta * A * B.block(0, col, N, out.cols() - col);
}
*/
#endif /* LINOP_H */
