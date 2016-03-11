#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "linop.h"
#include "chol.h"

TEST_CASE( "SparseFeat", "[At_mul_A]" ) {
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
