#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>

#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>

#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>

#include <omp.h>

#include "macauvb.h"
#include "mvnormal.h"
#include "bpmfutils.h"

using namespace std; 
using namespace Eigen;

void MacauVB::addPrior(std::unique_ptr<ILatentPriorVB> & prior) {
  priors.push_back( std::move(prior) );
}

void MacauVB::setPrecision(double p) {
  alpha = p;
}

void MacauVB::setNiter(int ni) {
  niter = ni;
}

void MacauVB::setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  Y.resize(nrows, ncols);
  sparseFromIJV(Y, rows, cols, values, N);
  Yt = Y.transpose();
  mean_rating = Y.sum() / Y.nonZeros();
}

void MacauVB::setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
  Ytest.resize(nrows, ncols);
  sparseFromIJV(Ytest, rows, cols, values, N);
}

void MacauVB::init() {
  unsigned seed1 = std::chrono::system_clock::now().time_since_epoch().count();
  if (priors.size() != 2) {
    throw std::runtime_error("Only 2 priors are supported.");
  }
  init_bmrng(seed1);
  MatrixXd* U = new MatrixXd(num_latent, Y.rows());
  MatrixXd* V = new MatrixXd(num_latent, Y.cols());
  bmrandn(*U);
  bmrandn(*V);
  (*U) *= latent_init_std;
  (*V) *= latent_init_std;
  samples.push_back( std::move(std::unique_ptr<MatrixXd>(U)) );
  samples.push_back( std::move(std::unique_ptr<MatrixXd>(V)) );
}

void MacauVB::run() {
  // TODO
}
