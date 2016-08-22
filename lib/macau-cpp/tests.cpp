#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "linop.h"
#include "chol.h"
#include "mvnormal.h"
#include "latentprior.h"
#include "latentpriorvb.h"
#include "bpmfutils.h"
#include <cmath>
#include <memory>

TEST_CASE( "SparseFeat/At_mul_A_bcsr", "[At_mul_A] for BinaryCSR" ) {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  SparseFeat sf(6, 4, 9, rows, cols);

  REQUIRE( sf.M.nrow == 6 );
  REQUIRE( sf.M.ncol == 4 );

  Eigen::MatrixXd AA(4, 4);
  At_mul_A(AA, sf);
  REQUIRE( AA(0,0) == 2 );
  REQUIRE( AA(1,1) == 3 );
  REQUIRE( AA(2,2) == 2 );
  REQUIRE( AA(3,3) == 2 );
  REQUIRE( AA(1,0) == 0 );
  REQUIRE( AA(2,0) == 2 );
  REQUIRE( AA(3,0) == 0 );

  REQUIRE( AA(2,1) == 0 );
  REQUIRE( AA(3,1) == 1 );

  REQUIRE( AA(3,2) == 0 );
}

TEST_CASE( "SparseFeat/At_mul_A_csr", "[At_mul_A] for CSR" ) {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
  SparseDoubleFeat sf(6, 4, 9, rows, cols, vals);

  REQUIRE( sf.M.nrow == 6 );
  REQUIRE( sf.M.ncol == 4 );

  Eigen::MatrixXd AA(4, 4);
  At_mul_A(AA, sf);
  REQUIRE( AA(0,0) == Approx(4.3801) );
  REQUIRE( AA(1,1) == Approx(2.4485) );
  REQUIRE( AA(2,2) == Approx(8.6420) );
  REQUIRE( AA(3,3) == Approx(5.9572) );

  REQUIRE( AA(1,0) == 0 );
  REQUIRE( AA(2,0) == Approx(3.8282) );
  REQUIRE( AA(3,0) == 0 );

  REQUIRE( AA(2,1) == 0 );
  REQUIRE( AA(3,1) == Approx(0.0714) );

  REQUIRE( AA(3,2) == 0 );
}

TEST_CASE( "linop/A_mul_Bx(csr)", "A_mul_Bx for CSR" ) {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
  SparseDoubleFeat sf(6, 4, 9, rows, cols, vals);
  Eigen::MatrixXd B(2, 4), X(2, 6), Xtr(2, 6);
  B << -1.38,  1.04, -0.28, -0.18,
        0.03,  0.88,  1.32, -0.31;
  Xtr << 0.624 , -0.8528,  1.2268,  0.6344, -3.4022, -0.4392,
         0.528 , -0.7216,  1.0286,  1.9308,  3.4113, -0.7564;
  A_mul_Bx<2>(X, sf.M,  B);
  REQUIRE( (X - Xtr).norm() == Approx(0) );
}

TEST_CASE( "linop/AtA_mul_Bx(csr)", "AtA_mul_Bx for CSR" ) {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
  SparseDoubleFeat sf(6, 4, 9, rows, cols, vals);
  Eigen::MatrixXd B(2, 4), tmp(2, 6), out(2, 4), outtr(2, 4), X(6, 4);
  B << -1.38,  1.04, -0.28, -0.18,
        0.03,  0.88,  1.32, -0.31;
  double reg = 0.6;

  X <<  0.  ,  0.6 ,  0.  ,  0.  ,
        0.  , -0.82,  0.  ,  0.  ,
        0.  ,  1.19,  0.  ,  0.06,
       -0.76,  0.  ,  1.48,  0.  ,
        1.95,  0.  ,  2.54,  0.  ,
        0.  ,  0.  ,  0.  ,  2.44;

  AtA_mul_Bx<2>(out, sf, reg, B, tmp);
  outtr = (X.transpose() * X * B.transpose() + reg * B.transpose()).transpose();
  REQUIRE( (out - outtr).norm() == Approx(0) );
}

TEST_CASE( "linop/AtA_mul_B_1thread(bcsr)", "AtA_mul_B_1thread for BinaryCSR" ) {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  SparseFeat sf(6, 4, 9, rows, cols);
  Eigen::VectorXd b(4), tmp(6), out(4), outtr(4);
  Eigen::MatrixXd X(6, 4);
  b << -1.38,  1.04, -0.28, -0.18;
  double reg = 0.6;

  X <<  0,  1,  0,  0,
        0,  1,  0,  0,
        0,  1,  0,  1,
        1,  0,  1,  0,
        1,  0,  1,  0,
        0,  0,  0,  1;

  AtA_mul_B_1thread(out, sf, reg, b, tmp);
  outtr = X.transpose() * X * b + reg * b;
  REQUIRE( (out - outtr).norm() == Approx(0) );
}

TEST_CASE( "linop/AtA_mul_B_1thread(csr)", "AtA_mul_B_1thread for CSR" ) {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
  SparseDoubleFeat sf(6, 4, 9, rows, cols, vals);
  Eigen::VectorXd b(4), tmp(6), out(4), outtr(4);
  Eigen::MatrixXd X(6, 4);
  b << -1.38,  1.04, -0.28, -0.18;
  double reg = 0.6;

  X <<  0.  ,  0.6 ,  0.  ,  0.  ,
        0.  , -0.82,  0.  ,  0.  ,
        0.  ,  1.19,  0.  ,  0.06,
       -0.76,  0.  ,  1.48,  0.  ,
        1.95,  0.  ,  2.54,  0.  ,
        0.  ,  0.  ,  0.  ,  2.44;

  AtA_mul_B_1thread(out, sf, reg, b, tmp);
  outtr = X.transpose() * X * b + reg * b;
  REQUIRE( (out - outtr).norm() == Approx(0) );
}

TEST_CASE( "linop/solve_blockcg_1thread(csr)", "solve_blockcg_1thread for CSR" ) {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
  SparseDoubleFeat sf(6, 4, 9, rows, cols, vals);
  Eigen::VectorXd b(4), x(4), xtr(4);
  Eigen::MatrixXd X(6, 4);
  b << -1.38,  1.04, -0.28, -0.18;
  double reg = 0.6;

  X <<  0.  ,  0.6 ,  0.  ,  0.  ,
        0.  , -0.82,  0.  ,  0.  ,
        0.  ,  1.19,  0.  ,  0.06,
       -0.76,  0.  ,  1.48,  0.  ,
        1.95,  0.  ,  2.54,  0.  ,
        0.  ,  0.  ,  0.  ,  2.44;

  x << 0, 0, 0, 0;
  solve_blockcg_1thread(x, sf, reg, b, 1e-6, 1e-6);
  Eigen::MatrixXd XX = X.transpose() * X + reg * Eigen::MatrixXd::Identity(4, 4);
  xtr = XX.colPivHouseholderQr().solve(b);
  REQUIRE( (x - xtr).norm() == Approx(0) );

  x << 0.5, 0.1, -0.8, 0.9;
  solve_blockcg_1thread(x, sf, reg, b, 1e-6, 1e-6);
  REQUIRE( (x - xtr).norm() == Approx(0) );
}

TEST_CASE( "linop/SparseFeat/col_square_sum", "sum of squares of a column" ) {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  SparseFeat sf(6, 4, 9, rows, cols);
  Eigen::VectorXd sq(4), sq_true(4);

  sq_true << 2, 3, 2, 2;
  sq = col_square_sum(sf);
  REQUIRE( sq(0) == Approx(sq_true(0)) );
  REQUIRE( sq(1) == Approx(sq_true(1)) );
  REQUIRE( sq(2) == Approx(sq_true(2)) );
  REQUIRE( sq(3) == Approx(sq_true(3)) );
}

TEST_CASE( "linop/SparseDoubleFeat/col_square_sum", "sum of squares of a column" ) {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
  SparseDoubleFeat sf(6, 4, 9, rows, cols, vals);
  Eigen::VectorXd sq(4), sq_true(4);

  sq_true << 4.3801,  2.4485,  8.642 ,  5.9572;
  sq = col_square_sum(sf);
  REQUIRE( sq(0) == Approx(sq_true(0)) );
  REQUIRE( sq(1) == Approx(sq_true(1)) );
  REQUIRE( sq(2) == Approx(sq_true(2)) );
  REQUIRE( sq(3) == Approx(sq_true(3)) );
}

TEST_CASE( "linop/SparseFeat/compute_uhat", "compute_uhat" ) {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  SparseFeat sf(6, 4, 9, rows, cols);
  Eigen::MatrixXd beta(3, 4), uhat(3, 6), uhat_true(3, 6);

  beta << 0.56,  0.55,  0.3 , -1.78,
          1.63, -0.71,  0.8 , -0.28,
          0.47,  0.37, -1.36,  0.86;
  uhat_true <<  0.55,  0.55, -1.23,  0.86,  0.86, -1.78,
               -0.71, -0.71, -0.99,  2.43,  2.43, -0.28,
                0.37,  0.37,  1.23, -0.89, -0.89,  0.86;

  compute_uhat(uhat, sf, beta);
  for (int i = 0; i < uhat.rows(); i++) {
    for (int j = 0; j < uhat.cols(); j++) {
      REQUIRE( uhat(i,j) == Approx(uhat_true(i,j)) );
    }
  }
}

TEST_CASE( "linop/SparseFeat/solve_blockcg", "BlockCG solver (1rhs)" ) {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  SparseFeat sf(6, 4, 9, rows, cols);
  Eigen::MatrixXd B(1, 4), X(1, 4), X_true(1, 4);

  B << 0.56,  0.55,  0.3 , -1.78;
  X_true << 0.35555556,  0.40709677, -0.16444444, -0.87483871;
  int niter = solve_blockcg(X, sf, 0.5, B, 1e-6);
  for (int i = 0; i < X.rows(); i++) {
    for (int j = 0; j < X.cols(); j++) {
      REQUIRE( X(i,j) == Approx(X_true(i,j)) );
    }
  }
  REQUIRE( niter <= 4);
}


TEST_CASE( "linop/SparseFeat/solve_blockcg_1_0", "BlockCG solver (3rhs separately)" ) {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  SparseFeat sf(6, 4, 9, rows, cols);
  Eigen::MatrixXd B(3, 4), X(3, 4), X_true(3, 4);

  B << 0.56,  0.55,  0.3 , -1.78,
       0.34,  0.05, -1.48,  1.11,
       0.09,  0.51, -0.63,  1.59;

  X_true << 0.35555556,  0.40709677, -0.16444444, -0.87483871,
            1.69333333, -0.12709677, -1.94666667,  0.49483871,
            0.66      , -0.04064516, -0.78      ,  0.65225806;

  solve_blockcg(X, sf, 0.5, B, 1e-6, 1, 0);
  for (int i = 0; i < X.rows(); i++) {
    for (int j = 0; j < X.cols(); j++) {
      REQUIRE( X(i,j) == Approx(X_true(i,j)) );
    }
  }
}

TEST_CASE( "linop/MatrixXd/compute_uhat", "compute_uhat for MatrixXd" ) {
  Eigen::MatrixXd beta(2, 4), feat(6, 4), uhat(2, 6), uhat_true(2, 6);
  beta << 0.56,  0.55,  0.3 , -1.78,
          1.63, -0.71,  0.8 , -0.28;
  feat <<  -0.83,  -0.26,  -0.52,  -0.27,
            0.91,  -0.48,   0.50,  -0.20,
           -0.59,   1.94,  -1.09,   0.86,
           -0.08,   0.62,  -1.10,   0.96,
            1.44,   0.89,  -0.45,   0.2,
           -1.33,  -1.42,   0.03,  -2.32;
  uhat_true <<  -0.2832,  0.7516,  -1.1212,  -1.7426,  0.8049,   2.6128,
                -1.5087,  2.2801,  -3.4519,  -1.7194,  1.2993,  -0.4861;
  compute_uhat(uhat, feat, beta);
  for (int i = 0; i < uhat.rows(); i++) {
    for (int j = 0; j < uhat.cols(); j++) {
      REQUIRE( uhat(i,j) == Approx(uhat_true(i,j)) );
    }
  }
}

TEST_CASE( "chol/chol_solve_t", "[chol_solve_t]" ) {
  Eigen::MatrixXd m(3,3), rhs(5,3), xopt(5,3);
  m << 7, 0, 0,
       2, 5, 0,
       6, 1, 6;

  rhs << -1.227, -0.890,  0.293,
          0.356, -0.733, -1.201,
         -0.003, -0.091, -1.467,
          0.819,  0.725, -0.719,
         -0.485,  0.955,  1.707;
  chol_decomp(m);
  chol_solve_t(m, rhs);
  xopt << -1.67161,  0.151609,  1.69517,
           2.10217, -0.545174, -2.21148,
           1.80587, -0.34187,  -1.99339,
           1.71883, -0.180826, -1.80852,
          -2.93874,  0.746739,  3.09878;

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 3; j++) {
      REQUIRE( rhs(i,j) == Approx(xopt(i,j)) );
    }
  }
}

TEST_CASE( "mvnormal/rgamma", "generaring random gamma variable" ) {
  init_bmrng(1234);
  double g = rgamma(100.0, 0.01);
  REQUIRE( g > 0 );
}

TEST_CASE( "latentprior/sample_lambda_beta", "sampling lambda beta from gamma distribution" ) {
  init_bmrng(1234);
  Eigen::MatrixXd beta(2, 3), Lambda_u(2, 2);
  beta << 3.0, -2.00,  0.5,
          1.0,  0.91, -0.2;
  Lambda_u << 0.5, 0.1,
              0.1, 0.3;
  auto post = posterior_lambda_beta(beta, Lambda_u, 0.01, 0.05);
  REQUIRE( post.first  == Approx(3.005) );
  REQUIRE( post.second == Approx(0.2631083888) );

  double lambda_beta = sample_lambda_beta(beta, Lambda_u, 0.01, 0.05);
  REQUIRE( lambda_beta > 0 );
}

TEST_CASE( "linop/A_mul_At_omp", "A_mul_At with OpenMP" ) {
  init_bmrng(12345);
  Eigen::MatrixXd A(2, 42);
  Eigen::MatrixXd AAt(2, 2);
  bmrandn(A);
  A_mul_At_omp(AAt, A);
  Eigen::MatrixXd AAt_true(2, 2);
  AAt_true.triangularView<Eigen::Lower>() = A * A.transpose();

  REQUIRE( AAt(0,0) == Approx(AAt_true(0,0)) );
  REQUIRE( AAt(1,1) == Approx(AAt_true(1,1)) );
  REQUIRE( AAt(1,0) == Approx(AAt_true(1,0)) );
}

TEST_CASE( "linop/A_mul_At_combo", "A_mul_At with OpenMP (returning matrix)" ) {
  init_bmrng(12345);
  Eigen::MatrixXd A(2, 42);
  Eigen::MatrixXd AAt_true(2, 2);
  bmrandn(A);
  Eigen::MatrixXd AAt = A_mul_At_combo(A);
  AAt_true = A * A.transpose();

  REQUIRE( AAt.rows() == 2);
  REQUIRE( AAt.cols() == 2);

  REQUIRE( AAt(0,0) == Approx(AAt_true(0,0)) );
  REQUIRE( AAt(1,1) == Approx(AAt_true(1,1)) );
  REQUIRE( AAt(0,1) == Approx(AAt_true(0,1)) );
  REQUIRE( AAt(1,0) == Approx(AAt_true(1,0)) );
}

TEST_CASE( "linop/A_mul_B_omp", "Fast parallel A_mul_B for small A") {
  Eigen::MatrixXd A(2, 2);
  Eigen::MatrixXd B(2, 5);
  Eigen::MatrixXd C(2, 5);
  Eigen::MatrixXd Ctr(2, 5);
  A << 3.0, -2.00,
       1.0,  0.91;
  B << 0.52, 0.19, 0.25, -0.73, -2.81,
      -0.15, 0.31,-0.40,  0.91, -0.08;
  A_mul_B_omp(0, C, 1.0, A, B);
  Ctr = A * B;
  REQUIRE( (C - Ctr).norm() == Approx(0.0) );
}

TEST_CASE( "linop/A_mul_B_omp/speed", "Speed of A_mul_B_omp") {
  Eigen::MatrixXd B(32, 1000);
  Eigen::MatrixXd X(32, 1000);
  Eigen::MatrixXd Xtr(32, 1000);
  Eigen::MatrixXd A(32, 32);
  for (int col = 0; col < B.cols(); col++) {
    for (int row = 0; row < B.rows(); row++) {
      B(row, col) = sin(row * col);
    }
  }
  for (int col = 0; col < A.cols(); col++) {
    for (int row = 0; row < A.rows(); row++) {
      A(row, col) = sin(row*(row+0.2)*col);
    }
  }
  Xtr = A * B;
  A_mul_B_omp(0, X, 1.0, A, B);
  REQUIRE( (X - Xtr).norm() == Approx(0.0) );
}

TEST_CASE( "linop/A_mul_B_add", "Fast parallel A_mul_B with adding") {
  Eigen::MatrixXd A(2, 2);
  Eigen::MatrixXd B(2, 5);
  Eigen::MatrixXd C(2, 5);
  Eigen::MatrixXd Ctr(2, 5);
  A << 3.0, -2.00,
       1.0,  0.91;
  B << 0.52, 0.19, 0.25, -0.73, -2.81,
      -0.15, 0.31,-0.40,  0.91, -0.08;
  C << 0.21, 0.70, 0.53, -0.18, -2.14,
      -0.35,-0.82,-0.27,  0.15, -0.10;
  Ctr = C;
  A_mul_B_omp(1.0, C, 1.0, A, B);
  Ctr += A * B;
  REQUIRE( (C - Ctr).norm() == Approx(0.0) );
}

TEST_CASE( "linop/At_mul_Bt/SparseFeat", "At_mul_Bt of single col for SparseFeat") {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  SparseFeat sf(6, 4, 9, rows, cols);
  Eigen::MatrixXd B(2, 6);
  Eigen::VectorXd Y(2), Y_true(2);
  B << -0.23, -2.89, -1.04, -0.52, -1.45, -1.42,
       -0.16, -0.62,  1.19,  1.12,  0.11,  0.61;
  Y_true << -4.16, 0.41;

  At_mul_Bt(Y, sf, 1, B);
  REQUIRE( Y(0) == Approx(Y_true(0)) );
  REQUIRE( Y(1) == Approx(Y_true(1)) );
}

TEST_CASE( "linop/At_mul_Bt/SparseDoubleFeat", "At_mul_Bt of single col for SparseDoubleFeat") {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
  SparseDoubleFeat sf(6, 4, 9, rows, cols, vals);

  Eigen::MatrixXd B(2, 6);
  Eigen::VectorXd Y(2), Y_true(2);
  B << -0.23, -2.89, -1.04, -0.52, -1.45, -1.42,
       -0.16, -0.62,  1.19,  1.12,  0.11,  0.61;
  Y_true << 0.9942,  1.8285;

  At_mul_Bt(Y, sf, 1, B);
  REQUIRE( Y(0) == Approx(Y_true(0)) );
  REQUIRE( Y(1) == Approx(Y_true(1)) );
}

TEST_CASE( "linop/add_Acol_mul_bt/SparseFeat", "add_Acol_mul_bt for SparseFeat") {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  SparseFeat sf(6, 4, 9, rows, cols);
  Eigen::MatrixXd Z(2, 6), Z_added(2, 6);
  Eigen::VectorXd b(2);
  Z << -0.23, -2.89, -1.04, -0.52, -1.45, -1.42,
       -0.16, -0.62,  1.19,  1.12,  0.11,  0.61;
  b << -4.16, 0.41;
  Z_added << -4.39, -7.05, -5.2 , -0.52, -1.45, -1.42,
              0.25, -0.21,  1.6 ,  1.12,  0.11,  0.61;

  add_Acol_mul_bt(Z, sf, 1, b);
  REQUIRE( (Z - Z_added).norm() == Approx(0.0) );
}

// computes Z += A[:,col] * b', where a and b are vectors
TEST_CASE( "linop/add_Acol_mul_bt/SparseDoubleFeat", "add_Acol_mul_bt for SparseDoubleFeat") {
  int rows[9] = { 0, 3, 3, 2, 5, 4, 1, 2, 4 };
  int cols[9] = { 1, 0, 2, 1, 3, 0, 1, 3, 2 };
  double vals[9] = { 0.6 , -0.76,  1.48,  1.19,  2.44,  1.95, -0.82,  0.06,  2.54 };
  SparseDoubleFeat sf(6, 4, 9, rows, cols, vals);

  Eigen::MatrixXd Z(2, 6), Z_added(2, 6);
  Eigen::VectorXd b(2);
  Z << -0.23, -2.89, -1.04, -0.52, -1.45, -1.42,
       -0.16, -0.62,  1.19,  1.12,  0.11,  0.61;
  b << -4.16, 0.41;
  Z_added << -2.726 ,  0.5212, -5.9904, -0.52  , -1.45  , -1.42,
              0.086 , -0.9562,  1.6779,  1.12  ,  0.11  ,  0.61;

  add_Acol_mul_bt(Z, sf, 1, b);
  REQUIRE( (Z - Z_added).norm() == Approx(0.0) );
}

TEST_CASE( "latentpriorvb/bpmfpriorvb/update_latents", "BPMFPriorVB update_latents") {
  BPMFPriorVB prior(2, 3.0);
  prior.mu_mean  << 0.3, -0.2;
  prior.mu_var   << 1.5,  1.7;
  prior.lambda_b << 0.6,  0.7;
  const double mean_value = 1.7;
  const double alpha      = 2.1;

  Eigen::SparseMatrix<double> Y(3, 4), Yt;
  Y.insert(0, 1) = 3.0;
  Y.insert(0, 2) = 2.8;
  Y.insert(1, 0) = 1.2;
  Y.makeCompressed();
  Yt = Y.transpose();

  Eigen::MatrixXd Vmean(2, 4);
  Eigen::MatrixXd Vvar(2, 4);
  Vmean << 1.4, 5.0, 0.6, -0.7,
          -0.5, 2.8,-1.3,  1.8;
  Vvar  << 2.2, 2.1, 1.7, 0.9,
           1.6, 1.8, 1.9, 0.7;
  Eigen::MatrixXd Umean(2, 3);
  Eigen::MatrixXd Uvar(2, 3);
  Umean << 0.9, -1.2, 0.7,
           1.2, -0.8, 1.9;
  Uvar  << 0.9, -1.2, 0.7,
           1.2, -0.8, 1.9;
  const double ad = prior.lambda_a0 + (1 + Y.rows()) / 2.0;
  const Eigen::VectorXd Elambda = Eigen::VectorXd::Constant(2, ad).cwiseQuotient( prior.lambda_b );
  const double Q00 = Elambda(0) + alpha * (Vmean(0,1)*Vmean(0,1) + Vmean(0,2)*Vmean(0,2) + Vvar(0, 1) + Vvar(0, 2));
  const double L00 = Elambda(0) * prior.mu_mean(0) + alpha*(
                  (Y.coeff(0, 1) - mean_value - Umean(1, 0) * Vmean(1, 1)) * Vmean(0, 1) +
                  (Y.coeff(0, 2) - mean_value - Umean(1, 0) * Vmean(1, 2)) * Vmean(0, 2)
               );
  const double Q01 = Elambda(1) + alpha * (Vmean(1,1)*Vmean(1,1) + Vmean(1,2)*Vmean(1,2) + Vvar(1, 1) + Vvar(1, 2));
  const double L01 = Elambda(1) * prior.mu_mean(1) + alpha*(
                  (Y.coeff(0, 1) - mean_value - (L00 / Q00) * Vmean(0, 1)) * Vmean(1, 1) +
                  (Y.coeff(0, 2) - mean_value - (L00 / Q00) * Vmean(0, 2)) * Vmean(1, 2)
               );

  const double Q10 = Elambda(0) + alpha * (Vmean(0,0)*Vmean(0,0) + Vvar(0, 0));
  const double L10 = Elambda(0) * prior.mu_mean(0) + alpha*(
                  (Y.coeff(1, 0) - mean_value - Umean(1, 1) * Vmean(1, 0)) * Vmean(0, 0)
               );
  const double Q11 = Elambda(1) + alpha * (Vmean(1,0)*Vmean(1,0) + Vvar(1, 0));
  const double L11 = Elambda(1) * prior.mu_mean(1) + alpha*(
                  (Y.coeff(1, 0) - mean_value - (L10 / Q10) * Vmean(0, 0)) * Vmean(1, 0)
               );

  const double Q20 = Elambda(0);
  const double L20 = Elambda(0) * prior.mu_mean(0);
  const double Q21 = Elambda(1);
  const double L21 = Elambda(1) * prior.mu_mean(1);

  prior.update_latents(Umean, Uvar, Yt, mean_value, Vmean, Vvar, alpha);

  REQUIRE( Umean(0, 0) == Approx(L00 / Q00) );
  REQUIRE( Uvar (0, 0) == Approx(1.0 / Q00) );
  REQUIRE( Umean(1, 0) == Approx(L01 / Q01) );
  REQUIRE( Uvar (1, 0) == Approx(1.0 / Q01) );

  REQUIRE( Umean(0, 1) == Approx(L10 / Q10) );
  REQUIRE( Uvar (0, 1) == Approx(1.0 / Q10) );
  REQUIRE( Umean(1, 1) == Approx(L11 / Q11) );
  REQUIRE( Uvar (1, 1) == Approx(1.0 / Q11) );

  REQUIRE( Umean(0, 2) == Approx(L20 / Q20) );
  REQUIRE( Uvar (0, 2) == Approx(1.0 / Q20) );
  REQUIRE( Umean(1, 2) == Approx(L21 / Q21) );
  REQUIRE( Uvar (1, 2) == Approx(1.0 / Q21) );
}

TEST_CASE( "latentpriorvb/bpmfpriorvb/update_prior", "BPMFPriorVB update_prior") {
  BPMFPriorVB prior(2, 3.0);
  prior.mu_mean  << 0.3, -0.2;
  prior.mu_var   << 1.5,  1.7;
  prior.lambda_b << 0.6,  0.7;
  const double mean_value = 1.7;
  const double alpha      = 2.1;

  Eigen::MatrixXd Umean(2, 3);
  Eigen::MatrixXd Uvar(2, 3);
  Umean << 0.9, -1.2, 0.7,
           1.2, -0.8, 1.9;
  Uvar  << 0.9, -1.2, 0.7,
           1.2, -0.8, 1.9;
  // mu_mean and mu_var
  const double ad = prior.lambda_a0 + (1 + Umean.cols()) / 2.0;
  Eigen::VectorXd Elambda = Eigen::VectorXd::Constant(2, ad).cwiseQuotient( prior.lambda_b );
  Eigen::VectorXd A = Elambda * (prior.b0 + Umean.cols());
  Eigen::VectorXd B = Elambda.cwiseProduct( Umean.rowwise().sum() );
  Eigen::VectorXd mu_mean = B.cwiseQuotient(A);
  Eigen::VectorXd mu_var  = A.cwiseInverse();

  // lambda_b
  Eigen::VectorXd lambda_b = Eigen::VectorXd::Constant(2, prior.lambda_b0);
  lambda_b += 0.5 * prior.b0 * (mu_mean.cwiseProduct(mu_mean) + mu_var);
  auto udiff = Umean.colwise() - mu_mean;
  lambda_b += 0.5 * (udiff.cwiseProduct(udiff).rowwise().sum() + Uvar.rowwise().sum() + Umean.cols() * mu_var);

  prior.update_prior(Umean, Uvar);
  REQUIRE( prior.mu_mean(0) == Approx(mu_mean(0)) );
  REQUIRE( prior.mu_mean(1) == Approx(mu_mean(1)) );
  REQUIRE( prior.mu_var(0)  == Approx(mu_var(0))  );
  REQUIRE( prior.mu_var(1)  == Approx(mu_var(1))  );

  REQUIRE( prior.lambda_b(0) == Approx(lambda_b(0)) );
  REQUIRE( prior.lambda_b(1) == Approx(lambda_b(1)) );
}

TEST_CASE( "latentpriorvb/macaupriorvb/update_prior", "MacauPriorVB update_prior") {
  int rows[5] = { 0, 0, 1, 2, 2 };
  int cols[5] = { 0, 1, 1, 2, 3 };
  std::unique_ptr<SparseFeat> sf = std::unique_ptr<SparseFeat>(new SparseFeat(3, 4, 5, rows, cols));
  Eigen::MatrixXd Fdense(3, 4);
  Fdense << 1.,  1.,  0.,  0.,
            0.,  1.,  0.,  0.,
            0.,  0.,  1.,  1.;

  MacauPriorVB<SparseFeat> prior(2, sf, 3.0);
  prior.mu_mean  << 0.3, -0.2;
  prior.mu_var   << 1.5,  1.7;
  prior.lambda_b << 0.6,  0.7;
  const double mean_value = 1.7;
  const double alpha      = 2.1;
  prior.beta <<  0.5, 1.4, 0.7, -0.3,
                -0.2, 0.6, 0.6, -0.5;
  prior.beta_var <<  0.5, 1.4, 0.9, 0.8,
                     0.7, 0.3, 0.4, 0.5;
  prior.lambda_beta << 0.9/0.7, 0.4/0.6;
  prior.Uhat_valid = false;

  Eigen::SparseMatrix<double> Y(3, 4), Yt;
  Y.insert(0, 1) = 3.0;
  Y.insert(0, 2) = 2.8;
  Y.insert(1, 0) = 1.2;
  Y.makeCompressed();
  Yt = Y.transpose();

  Eigen::MatrixXd Umean(2, 3);
  Eigen::MatrixXd Uvar(2, 3);
  Umean << 0.9, -1.2, 0.7,
           1.2, -0.8, 1.9;
  Uvar  << 0.9, -1.2, 0.7,
           1.2, -0.8, 1.9;
  Eigen::MatrixXd Uhat(2, 3);
  Uhat = prior.beta * Fdense.transpose();

  // mu_mean and mu_var
  const double ad = prior.lambda_a0 + (1 + Umean.cols()) / 2.0;
  Eigen::VectorXd Elambda = Eigen::VectorXd::Constant(2, ad).cwiseQuotient( prior.lambda_b );
  Eigen::VectorXd A = Elambda * (prior.b0 + Umean.cols());
  Eigen::VectorXd B = Elambda.cwiseProduct( Umean.rowwise().sum() - Uhat.rowwise().sum());
  Eigen::VectorXd mu_mean = B.cwiseQuotient(A);
  Eigen::VectorXd mu_var  = A.cwiseInverse();

  // lambda_b
  Eigen::VectorXd lambda_b = Eigen::VectorXd::Constant(2, prior.lambda_b0);
  lambda_b += 0.5 * prior.b0 * (mu_mean.cwiseProduct(mu_mean) + mu_var);
  for (int d = 0; d < 2; d++) {
    for (int i = 0; i < Fdense.rows(); i++) {
      double du = Umean(d,i) - Uhat(d,i) - mu_mean(d);
      lambda_b(d) += 0.5 * (du * du + Uvar(d, i) + mu_var(d));
    }
  }

  for (int d = 0; d < 2; d++) {
    for (int i = 0; i < Fdense.rows(); i++) {
      for (int f = 0; f < Fdense.cols(); f++) {
        lambda_b[d] += 0.5 * prior.beta_var(d, f) * Fdense(i, f) * Fdense(i, f);
      }
    }
  }

  prior.update_prior(Umean, Uvar);

  REQUIRE( (prior.Uhat - Uhat).norm() == Approx(0.0) );
  REQUIRE( prior.mu_mean(0) == Approx(mu_mean(0)) );
  REQUIRE( prior.mu_mean(1) == Approx(mu_mean(1)) );
  REQUIRE( prior.mu_var(0)  == Approx(mu_var(0))  );
  REQUIRE( prior.mu_var(1)  == Approx(mu_var(1))  );

  REQUIRE( prior.lambda_b(0) == Approx(lambda_b(0)) );
  REQUIRE( prior.lambda_b(1) == Approx(lambda_b(1)) );
}

/*
TEST_CASE( "latentpriorvb/macaupriorvb/update_beta_uni", "MacauPriorVB update_beta") {
  int rows[5] = { 0, 0, 1, 2, 2 };
  int cols[5] = { 0, 1, 1, 2, 3 };
  std::unique_ptr<SparseFeat> sf = std::unique_ptr<SparseFeat>(new SparseFeat(3, 4, 5, rows, cols));
  Eigen::MatrixXd Fdense(3, 4);
  Fdense << 1.,  1.,  0.,  0.,
            0.,  1.,  0.,  0.,
            0.,  0.,  1.,  1.;

  MacauPriorVB<SparseFeat> prior(2, sf, 3.0);
  prior.mu_mean  << 0.3, -0.2;
  prior.lambda_b << 0.6,  0.7;
  prior.beta <<  0.5, 1.4, 0.7, -0.3,
                -0.2, 0.6, 0.6, -0.5;
  prior.beta_var <<  0.5, 1.4, 0.9, 0.8,
                     0.7, 0.3, 0.4, 0.5;
  prior.lambda_beta << 0.9 / 0.7, 0.4 / 0.6;

  Eigen::MatrixXd Umean(2, 3);
  Umean << 0.9, -1.2, 0.7,
           1.2, -0.8, 1.9;
  //Eigen::MatrixXd Uhat(2, 3);
  //Uhat = prior.beta * Fdense.transpose();
  //prior.Uhat = Uhat;

  // beta and beta_var
  Eigen::VectorXd Fsq = Fdense.colwise().sum();
  double ad = prior.lambda_a0 + (1 + Umean.cols()) / 2.0;
  Eigen::VectorXd E_lambda = Eigen::VectorXd::Constant(2, ad).cwiseQuotient( prior.lambda_b );
  Eigen::VectorXd E_lambda_beta = prior.lambda_beta;

  Eigen::MatrixXd beta     = prior.beta;
  Eigen::MatrixXd beta_var = prior.beta_var;
  for (int f = 0; f < Fdense.cols(); f++) {
    Eigen::VectorXd Adf = E_lambda_beta + E_lambda * Fsq(f);
    for (int d = 0; d < Umean.rows(); d++) {
      beta_var(d,f) = 1.0 / Adf(d);
    }
  }

  prior.update_beta_uni(Umean);

  REQUIRE( prior.beta_var(0, 0) == Approx(beta_var(0, 0)) );
  REQUIRE( prior.beta_var(1, 0) == Approx(beta_var(1, 0)) );

  REQUIRE( prior.beta(0, 0) == Approx(beta(0, 0)) );
  REQUIRE( prior.beta(1, 0) == Approx(beta(1, 0)) );

  REQUIRE( (prior.beta_var - beta_var).norm() == Approx(0.0) );
  REQUIRE( (prior.beta - beta).norm() == Approx(0.0) );
  REQUIRE( prior.Uhat_valid == false );
}*/

TEST_CASE( "latentpriorvb/macaupriorvb/update_beta_uni", "MacauPriorVB update_beta_uni") {
  int rows[5] = { 0, 0, 1, 2, 2 };
  int cols[5] = { 0, 1, 1, 2, 3 };
  std::unique_ptr<SparseFeat> sf = std::unique_ptr<SparseFeat>(new SparseFeat(3, 4, 5, rows, cols));
  Eigen::MatrixXd Fdense(3, 4);
  Fdense << 1.,  1.,  0.,  0.,
            0.,  1.,  0.,  0.,
            0.,  0.,  1.,  1.;

  MacauPriorVB<SparseFeat> prior(2, sf, 3.0);
  prior.mu_mean  << 0.3, -0.2;
  prior.lambda_b << 0.6,  0.7;
  prior.beta <<  0.5, 1.4, 0.7, -0.3,
                -0.2, 0.6, 0.6, -0.5;
  prior.beta_var <<  0.5, 1.4, 0.9, 0.8,
                     0.7, 0.3, 0.4, 0.5;
  prior.lambda_beta << 0.9 / 0.7, 0.4 / 0.6;

  Eigen::MatrixXd Umean(2, 3);
  Umean << 0.9, -1.2, 0.7,
           1.2, -0.8, 1.9;
  Eigen::MatrixXd Uhat(2, 3);
  Uhat = prior.beta * Fdense.transpose();
  prior.Uhat = Uhat;

  // beta and beta_var
  Eigen::VectorXd Fsq = Fdense.colwise().sum();
  double ad = prior.lambda_a0 + (1 + Umean.cols()) / 2.0;
  Eigen::VectorXd E_lambda = Eigen::VectorXd::Constant(2, ad).cwiseQuotient( prior.lambda_b );
  Eigen::VectorXd E_lambda_beta = prior.lambda_beta;

  Eigen::MatrixXd beta     = prior.beta;
  Eigen::MatrixXd beta_var = prior.beta_var;
  for (int f = 0; f < Fdense.cols(); f++) {
    Eigen::VectorXd Adf = E_lambda_beta + E_lambda * Fsq(f);
    Eigen::VectorXd Bdf(2);
    for (int d = 0; d < 2; d++) {
      double tmp = 0;
      for (int i = 0; i < Fdense.rows(); i++) {
        double diff = Umean(d, i) - prior.mu_mean(d);
        for (int e = 0; e < Fdense.cols(); e++) {
          if (e == f) continue;
          diff -= Fdense(i, e) * beta(d, e);
        }
        tmp += diff * Fdense(i, f);
      }
      Bdf(d)        = tmp * E_lambda(d);
      beta(d, f)    = Bdf(d) / Adf(d);
      beta_var(d,f) = 1.0 / Adf(d);
    }
  }

  prior.update_beta_uni(Umean);

  REQUIRE( prior.beta_var(0, 0) == Approx(beta_var(0, 0)) );
  REQUIRE( prior.beta_var(1, 0) == Approx(beta_var(1, 0)) );

  REQUIRE( prior.beta(0, 0) == Approx(beta(0, 0)) );
  REQUIRE( prior.beta(1, 0) == Approx(beta(1, 0)) );

  REQUIRE( (prior.beta_var - beta_var).norm() == Approx(0.0) );
  REQUIRE( (prior.beta - beta).norm() == Approx(0.0) );
  REQUIRE( prior.Uhat_valid == false );
}

TEST_CASE( "latentpriorvb/macaupriorvb/update_lambda_beta", "update_lambda_beta") {
  int rows[5] = { 0, 0, 1, 2, 2 };
  int cols[5] = { 0, 1, 1, 2, 3 };
  std::unique_ptr<SparseFeat> sf = std::unique_ptr<SparseFeat>(new SparseFeat(3, 4, 5, rows, cols));
  MacauPriorVB<SparseFeat> prior(2, sf, 3.0);
  prior.beta <<  0.5, 1.4, 0.7, -0.3,
                -0.2, 0.6, 0.6, -0.5;
  prior.beta_var <<  0.5, 1.4, 0.9, 0.8,
                     0.7, 0.3, 0.4, 0.5;
  prior.lambda_beta_a0 = 0.17;
  prior.lambda_beta_b0 = 0.11;
  prior.Uhat_valid = false;

  Eigen::VectorXd lambda_beta_a(2);
  Eigen::VectorXd lambda_beta_b(2);
  lambda_beta_a = Eigen::VectorXd::Constant(2, prior.lambda_beta_a0 + prior.beta.cols() / 2.0);
  lambda_beta_b = Eigen::VectorXd::Constant(2, prior.lambda_beta_b0) + prior.beta.cwiseProduct(prior.beta).rowwise().sum() / 2.0 + prior.beta_var.rowwise().sum() / 2.0;
  Eigen::VectorXd lambda_beta = lambda_beta_a.cwiseQuotient(lambda_beta_b);

  prior.update_lambda_beta();

  REQUIRE( prior.lambda_beta(0) == Approx(lambda_beta(0)) );
  REQUIRE( prior.lambda_beta(1) == Approx(lambda_beta(1)) );
}

TEST_CASE( "latentpriorvb/macaupriorvb/update_latents", "MacauPriorVB update_latents") {
  int rows[5] = { 0, 0, 1, 2, 2 };
  int cols[5] = { 0, 1, 1, 2, 3 };
  std::unique_ptr<SparseFeat> sf = std::unique_ptr<SparseFeat>(new SparseFeat(3, 4, 5, rows, cols));

  // for testing
  Eigen::MatrixXd Fdense(3, 4);
  Fdense << 1.,  1.,  0.,  0.,
            0.,  1.,  0.,  0.,
            0.,  0.,  1.,  1.;

  MacauPriorVB<SparseFeat> prior(2, sf, 3.0);
  prior.mu_mean  << 0.3, -0.2;
  prior.mu_var   << 1.5,  1.7;
  prior.lambda_b << 0.6,  0.7;
  const double mean_value = 1.7;
  const double alpha      = 2.1;
  REQUIRE( prior.beta.rows() == 2 );
  REQUIRE( prior.beta.cols() == 4 );
  prior.beta <<  0.5, 1.4, 0.7, -0.3,
                -0.2, 0.6, 0.6, -0.5;
  prior.Uhat_valid = false;

  Eigen::MatrixXd Uhat(2, 3);
  Uhat = prior.beta * Fdense.transpose();

  Eigen::SparseMatrix<double> Y(3, 4), Yt;
  Y.insert(0, 1) = 3.0;
  Y.insert(0, 2) = 2.8;
  Y.insert(1, 0) = 1.2;
  Y.makeCompressed();
  Yt = Y.transpose();

  Eigen::MatrixXd Vmean(2, 4);
  Eigen::MatrixXd Vvar(2, 4);
  Vmean << 1.4, 5.0, 0.6, -0.7,
          -0.5, 2.8,-1.3,  1.8;
  Vvar  << 2.2, 2.1, 1.7, 0.9,
           1.6, 1.8, 1.9, 0.7;
  Eigen::MatrixXd Umean(2, 3);
  Eigen::MatrixXd Uvar(2, 3);
  Umean << 0.9, -1.2, 0.7,
           1.2, -0.8, 1.9;
  Uvar  << 0.9, -1.2, 0.7,
           1.2, -0.8, 1.9;

  const double ad = prior.lambda_a0 + (1 + Y.rows()) / 2.0;
  const Eigen::VectorXd Elambda = Eigen::VectorXd::Constant(2, ad).cwiseQuotient( prior.lambda_b );
  const double Q00 = Elambda(0) + alpha * (Vmean(0,1)*Vmean(0,1) + Vmean(0,2)*Vmean(0,2) + Vvar(0, 1) + Vvar(0, 2));
  const double L00 = Elambda(0) * (prior.mu_mean(0) + Uhat(0, 0)) + alpha*(
                  (Y.coeff(0, 1) - mean_value - Umean(1, 0) * Vmean(1, 1)) * Vmean(0, 1) +
                  (Y.coeff(0, 2) - mean_value - Umean(1, 0) * Vmean(1, 2)) * Vmean(0, 2)
               );
  const double Q01 = Elambda(1) + alpha * (Vmean(1,1)*Vmean(1,1) + Vmean(1,2)*Vmean(1,2) + Vvar(1, 1) + Vvar(1, 2));
  const double L01 = Elambda(1) * (prior.mu_mean(1) + Uhat(1, 0)) + alpha*(
                  (Y.coeff(0, 1) - mean_value - (L00 / Q00) * Vmean(0, 1)) * Vmean(1, 1) +
                  (Y.coeff(0, 2) - mean_value - (L00 / Q00) * Vmean(0, 2)) * Vmean(1, 2)
               );

  const double Q10 = Elambda(0) + alpha * (Vmean(0,0)*Vmean(0,0) + Vvar(0, 0));
  const double L10 = Elambda(0) * (prior.mu_mean(0) + Uhat(0, 1)) + alpha*(
                  (Y.coeff(1, 0) - mean_value - Umean(1, 1) * Vmean(1, 0)) * Vmean(0, 0)
               );
  const double Q11 = Elambda(1) + alpha * (Vmean(1,0)*Vmean(1,0) + Vvar(1, 0));
  const double L11 = Elambda(1) * (prior.mu_mean(1) + Uhat(1, 1)) + alpha*(
                  (Y.coeff(1, 0) - mean_value - (L10 / Q10) * Vmean(0, 0)) * Vmean(1, 0)
               );

  const double Q20 = Elambda(0);
  const double L20 = Elambda(0) * (prior.mu_mean(0) + Uhat(0, 2));
  const double Q21 = Elambda(1);
  const double L21 = Elambda(1) * (prior.mu_mean(1) + Uhat(1, 2));

  prior.update_latents(Umean, Uvar, Yt, mean_value, Vmean, Vvar, alpha);

  REQUIRE( Umean(0, 0) == Approx(L00 / Q00) );
  REQUIRE( Uvar (0, 0) == Approx(1.0 / Q00) );
  REQUIRE( Umean(1, 0) == Approx(L01 / Q01) );
  REQUIRE( Uvar (1, 0) == Approx(1.0 / Q01) );

  REQUIRE( Umean(0, 1) == Approx(L10 / Q10) );
  REQUIRE( Uvar (0, 1) == Approx(1.0 / Q10) );
  REQUIRE( Umean(1, 1) == Approx(L11 / Q11) );
  REQUIRE( Uvar (1, 1) == Approx(1.0 / Q11) );

  REQUIRE( Umean(0, 2) == Approx(L20 / Q20) );
  REQUIRE( Uvar (0, 2) == Approx(1.0 / Q20) );
  REQUIRE( Umean(1, 2) == Approx(L21 / Q21) );
  REQUIRE( Uvar (1, 2) == Approx(1.0 / Q21) );
}

TEST_CASE( "linop/At_mul_A_blas", "A'A with BLAS is correct") {
  Eigen::MatrixXd A(3, 2);
  Eigen::MatrixXd AtA(2, 2);
  Eigen::MatrixXd AtAtr(2, 2);
  A <<  1.7, -3.1,
        0.7,  2.9,
       -1.3,  1.5;
  AtAtr = A.transpose() * A;
  At_mul_A_blas(A, AtA.data());
  makeSymmetric(AtA);
  REQUIRE( (AtA - AtAtr).norm() == Approx(0.0) );
}

TEST_CASE( "linop/A_mul_At_blas", "AA' with BLAS is correct") {
  Eigen::MatrixXd A(3, 2);
  Eigen::MatrixXd AA(3, 3);
  Eigen::MatrixXd AAtr(3, 3);
  A <<  1.7, -3.1,
        0.7,  2.9,
       -1.3,  1.5;
  AAtr = A * A.transpose();
  A_mul_At_blas(A, AA.data());
  makeSymmetric(AA);
  REQUIRE( (AA - AAtr).norm() == Approx(0.0) );
}

TEST_CASE( "linop/A_mul_B_blas", "A_mul_B_blas is correct") {
  Eigen::MatrixXd A(3, 2);
  Eigen::MatrixXd B(2, 5);
  Eigen::MatrixXd C(3, 5);
  Eigen::MatrixXd Ctr(3, 5);
  A << 3.0, -2.00,
       1.0,  0.91,
       1.9, -1.82;
  B << 0.52, 0.19, 0.25, -0.73, -2.81,
      -0.15, 0.31,-0.40,  0.91, -0.08;
  C << 0.21, 0.70, 0.53, -0.18, -2.14,
      -0.35,-0.82,-0.27,  0.15, -0.10,
      +2.34,-0.81,-0.47,  0.31, -0.14;
  A_mul_B_blas(C, A, B);
  Ctr = A * B;
  REQUIRE( (C - Ctr).norm() == Approx(0.0) );
}

TEST_CASE( "linop/At_mul_B_blas", "At_mul_B_blas is correct") {
  Eigen::MatrixXd A(2, 3);
  Eigen::MatrixXd B(2, 5);
  Eigen::MatrixXd C(3, 5);
  Eigen::MatrixXd Ctr(3, 5);
  A << 3.0, -2.00,  1.0,
       0.91, 1.90, -1.82;
  B << 0.52, 0.19, 0.25, -0.73, -2.81,
      -0.15, 0.31,-0.40,  0.91, -0.08;
  C << 0.21, 0.70, 0.53, -0.18, -2.14,
      -0.35,-0.82,-0.27,  0.15, -0.10,
      +2.34,-0.81,-0.47,  0.31, -0.14;
  Ctr = C;
  At_mul_B_blas(C, A, B);
  Ctr = A.transpose() * B;
  REQUIRE( (C - Ctr).norm() == Approx(0.0) );
}

TEST_CASE( "linop/A_mul_Bt_blas", "A_mul_Bt_blas is correct") {
  Eigen::MatrixXd A(3, 2);
  Eigen::MatrixXd B(5, 2);
  Eigen::MatrixXd C(3, 5);
  Eigen::MatrixXd Ctr(3, 5);
  A << 3.0, -2.00,
       1.0,  0.91,
       1.9, -1.82;
  B << 0.52,  0.19,
       0.25, -0.73,
      -2.81, -0.15,
       0.31, -0.40,
       0.91, -0.08;
  C << 0.21, 0.70, 0.53, -0.18, -2.14,
      -0.35,-0.82,-0.27,  0.15, -0.10,
      +2.34,-0.81,-0.47,  0.31, -0.14;
  A_mul_Bt_blas(C, A, B);
  Ctr = A * B.transpose();
  REQUIRE( (C - Ctr).norm() == Approx(0.0) );
}

TEST_CASE( "bpmfutils/split_work_mpi", "Test if work splitting is correct") {
   int work3[3], work5[5];
   split_work_mpi(96, 3, work3);
   REQUIRE( work3[0] == 32 );
   REQUIRE( work3[1] == 32 );
   REQUIRE( work3[2] == 32 );

   split_work_mpi(97, 3, work3);
   REQUIRE( work3[0] == 33 );
   REQUIRE( work3[1] == 32 );
   REQUIRE( work3[2] == 32 );

   split_work_mpi(95, 3, work3);
   REQUIRE( work3[0] == 32 );
   REQUIRE( work3[1] == 32 );
   REQUIRE( work3[2] == 31 );

   split_work_mpi(80, 3, work3);
   REQUIRE( work3[0] == 28 );
   REQUIRE( work3[1] == 26 );
   REQUIRE( work3[2] == 26 );

   split_work_mpi(11, 5, work5);
   REQUIRE( work5[0] == 3 );
   REQUIRE( work5[1] == 2 );
   REQUIRE( work5[2] == 2 );
   REQUIRE( work5[3] == 2 );
   REQUIRE( work5[4] == 2 );
}

TEST_CASE( "bpmfutils/sparseFromIJV", "Convert triplets to Eigen SparseMatrix") {
  int rows[3] = {0, 1, 2};
  int cols[3] = {2, 1, 0};
  double vals[3] = {1.0, 0.0, 2.0};
  Eigen::SparseMatrix<double> Y;
  Y.resize(3, 3);

  sparseFromIJV(Y, rows, cols, vals, 3);
  REQUIRE( Y.nonZeros() == 3 );
}

TEST_CASE( "bpmfutils/eval_rmse", "Test if prediction variance is correctly calculated") {
  int rows[1] = {0};
  int cols[1] = {0};
  double vals[1] = {4.5};
  Eigen::SparseMatrix<double> Y;
  Y.resize(1, 1);
  sparseFromIJV(Y, rows, cols, vals, 1);
  double mean_value = 2.0;

  Eigen::VectorXd pred     = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd pred_var = Eigen::VectorXd::Zero(1);
  Eigen::MatrixXd U(2, 1), V(2, 1);

  // first iteration
  U << 1.0, 0.0;
  V << 1.0, 0.0;
  auto rmse0 = eval_rmse(Y, 0, pred, pred_var, U, V, mean_value);
  REQUIRE(pred(0)      == Approx(3.0));
  REQUIRE(pred_var(0)  == Approx(0.0));
  REQUIRE(rmse0.first  == Approx(1.5));
  REQUIRE(rmse0.second == Approx(1.5));

  //// second iteration
  U << 2.0, 0.0;
  V << 1.0, 0.0;
  auto rmse1 = eval_rmse(Y, 1, pred, pred_var, U, V, mean_value);
  REQUIRE(pred(0)      == Approx((3.0 + 4.0) / 2));
  REQUIRE(pred_var(0)  == Approx(0.5));
  REQUIRE(rmse1.first  == 0.5);
  REQUIRE(rmse1.second == 1.0);

  //// third iteration
  U << 2.0, 0.0;
  V << 3.0, 0.0;
  auto rmse2 = eval_rmse(Y, 2, pred, pred_var, U, V, mean_value);
  REQUIRE(pred(0)      == Approx((3.0 + 4.0 + 8.0) / 3));
  REQUIRE(pred_var(0)  == Approx(14.0)); // accumulated variance
  REQUIRE(rmse2.first  == 3.5);
  REQUIRE(rmse2.second == 0.5);
}

TEST_CASE( "bpmfutils/row_mean_var", "Test if row_mean_var is correct") {
  Eigen::VectorXd mean(3), var(3), mean_tr(3), var_tr(3);
  Eigen::MatrixXd C(3, 5);
  C << 0.21, 0.70, 0.53, -0.18, -2.14,
      -0.35,-0.82,-0.27,  0.15, -0.10,
      +2.34,-0.81,-0.47,  0.31, -0.14;
  row_mean_var(mean, var, C);
  mean_tr = C.rowwise().mean();
  var_tr  = (C.colwise() - mean).cwiseAbs2().rowwise().mean();
  REQUIRE( (mean - mean_tr).norm() == Approx(0.0) );
  REQUIRE( (var  - var_tr).norm()  == Approx(0.0) );
}
