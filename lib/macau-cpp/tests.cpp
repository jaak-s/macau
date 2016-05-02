#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "linop.h"
#include "chol.h"
#include "mvnormal.h"
#include "latentprior.h"
#include "latentpriorvb.h"
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
  auto udiff = (Umean - Uhat).colwise() - mu_mean;
  lambda_b += 0.5 * (udiff.cwiseProduct(udiff).rowwise().sum() + Uvar.rowwise().sum() + Umean.cols() * mu_var);
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

TEST_CASE( "latentpriorvb/macaupriorvb/update_latents", "MacauPriorVB update_latents") {
  int rows[5] = { 0, 0, 1, 2, 2 };
  int cols[5] = { 0, 1, 1, 2, 3 };
  std::unique_ptr<SparseFeat> sf = std::unique_ptr<SparseFeat>(new SparseFeat(3, 4, 5, rows, cols));

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
  /*
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
  */
}
