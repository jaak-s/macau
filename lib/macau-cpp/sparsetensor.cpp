#include <Eigen/Dense>
#include <memory>
#include <iostream>
#include "sparsetensor.h"
#include "latentprior.h"
#include "noisemodels.h"

using namespace Eigen;


void SparseMode::init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int m, int mode_size) {
  if (idx.rows() != vals.size()) {
    throw std::runtime_error("idx.rows() must equal vals.size()");
  }
  num_modes = idx.cols();
  row_ptr.resize(mode_size + 1);
  row_ptr.setZero();
  values.resize(vals.size());
  indices.resize(idx.rows(), idx.cols() - 1);

  mode = m;
  auto rows = idx.col(mode);
  const int nrow = mode_size;
  nnz  = idx.rows();

  // compute number of non-zero entries per each element for the mode
  for (int i = 0; i < nnz; i++) {
    if (rows(i) >= mode_size) {
      throw std::runtime_error("SparseMode: mode value larger than mode_size");
    }
    row_ptr(rows(i))++;
  }
  // cumsum counts
  for (int row = 0, cumsum = 0; row < nrow; row++) {
    int temp     = row_ptr(row);
    row_ptr(row) = cumsum;
    cumsum      += temp;
  }
  row_ptr(nrow) = nnz;

  // writing idx and vals to indices and values
  for (int i = 0; i < nnz; i++) {
    int row  = rows(i);
    int dest = row_ptr(row);
    for (int j = 0, nj = 0; j < idx.cols(); j++) {
      if (j == mode) continue;
      indices(dest, nj) = idx(i, j);
      nj++;
    }
    //A->cols[dest] = cols[i];
    values(dest) = vals(i);
    row_ptr(row)++;
  }
  // fixing row_ptr
  for (int row = 0, prev = 0; row <= nrow; row++) {
    int temp     = row_ptr(row);
    row_ptr(row) = prev;
    prev         = temp;
  }
}


Eigen::MatrixXd MatrixData::getTestData() {
  MatrixXd coords( getTestNonzeros(), 3);
#pragma omp parallel for schedule(dynamic, 2)
  for (int k = 0; k < Ytest.outerSize(); ++k) {
    int idx = Ytest.outerIndexPtr()[k];
    for (SparseMatrix<double>::InnerIterator it(Ytest,k); it; ++it) {
      coords(idx, 0) = it.row();
      coords(idx, 1) = it.col();
      coords(idx, 2) = it.value();
      idx++;
    }
  }
  return coords;
}

void MatrixData::setTrain(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) {
	Y.resize(nrows, ncols);
	sparseFromIJV(Y, rows, cols, values, nnz);
	Yt = Y.transpose();
	mean_value = Y.sum() / Y.nonZeros();
	dims.resize(2);
	dims << Y.rows(), Y.cols();
}

void MatrixData::setTest(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) {
	Ytest.resize(nrows, ncols);
	sparseFromIJV(Ytest, rows, cols, values, nnz);
}

void MatrixData::setTrain(int* columns, int nmodes, double* values, int nnz, int* d) {
  if (nmodes != 2) {
    throw std::runtime_error("MatrixData: tensor training input not supported.");
  }
  auto idx  = toMatrix(columns, nnz, nmodes);
  auto vals = toVector(values, nnz);

  Y.resize(d[0], d[1]);
  sparseFromIJV(Y, idx, vals);

  Yt = Y.transpose();
  mean_value = Y.sum() / Y.nonZeros();
  dims.resize(2);
  dims << Y.rows(), Y.cols();
}
    
void MatrixData::setTest(int* columns, int nmodes, double* values, int nnz, int* d) {
  if (nmodes != 2) {
    throw std::runtime_error("MatrixData: tensor training input not supported.");
  }
  auto idx  = toMatrix(columns, nnz, nmodes);
  auto vals = toVector(values, nnz);

  Ytest.resize(d[0], d[1]);
  sparseFromIJV(Ytest, idx, vals);
}


////////  TensorData  ///////

void TensorData::setTrain(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi &d) {
  if (idx.rows() != vals.size()) {
    throw std::runtime_error("setTrain(): idx.rows() must equal vals.size()");
  }
  dims = d;
	mean_value = vals.mean();
  for (int mode = 0; mode < N; mode++) {
    Y->push_back( std::unique_ptr<SparseMode>(new SparseMode(idx, vals, mode, dims(mode))) );
  }
}

void TensorData::setTrain(int* columns, int nmodes, double* values, int nnz, int* dims) {
  auto idx  = toMatrix(columns, nnz, nmodes);
  auto vals = toVector(values, nnz);
  auto d    = toVector(dims, nmodes);
  setTrain(idx, vals, d);
}
    
void TensorData::setTest(int* columns, int nmodes, double* values, int nnz, int* dims) {
  auto idx  = toMatrix(columns, nnz, nmodes);
  auto vals = toVector(values, nnz);
  auto d    = toVector(dims, nmodes);
  setTest(idx, vals, d);
}


void TensorData::setTest(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi &d) {
  if (idx.rows() != vals.size()) {
    throw std::runtime_error("setTest(): idx.rows() must equal vals.size()");
  }
  if ((d - dims).norm() != 0) {
    throw std::runtime_error("setTest(): train and test Tensor sizes are not equal.");
  }
  Ytest = SparseMode(idx, vals, 0, d(0));
}

void TensorData::setTrain(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) {
  auto idx  = toMatrix(rows, cols, nnz);
  auto vals = toVector(values, nnz);

  VectorXi d(2);
  d << nrows, ncols;

  setTrain(idx, vals, d);
}

void TensorData::setTest(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) {
  auto idx  = toMatrix(rows, cols, nnz);
  auto vals = toVector(values, nnz);

  VectorXi d(2);
  d << nrows, ncols;

  setTest(idx, vals, d);
}

Eigen::MatrixXd TensorData::getTestData() {
  MatrixXd coords( getTestNonzeros(), N + 1);
#pragma omp parallel for schedule(dynamic, 2)
  for (int k = 0; k < Ytest.modeSize(); ++k) {
    for (int idx = Ytest.row_ptr[k]; idx < Ytest.row_ptr[k+1]; idx++) {
      coords(idx, 0) = (double)k;
      for (int col = 0; col < Ytest.indices.cols(); col++) {
        coords(idx, col + 1) = (double)Ytest.indices(idx, col);
      }
      coords(idx, N) = Ytest.values(idx);
    }
  }
  return coords;
}

// util functions
Eigen::MatrixXi toMatrix(int* col1, int* col2, int nrows) {
  Eigen::MatrixXi idx(nrows, 2);
  for (int row = 0; row < nrows; row++) {
    idx(row, 0) = col1[row];
    idx(row, 1) = col2[row];
  }
  return idx;
}

Eigen::MatrixXi toMatrix(int* columns, int nrows, int ncols) {
  Eigen::MatrixXi idx(nrows, ncols);
  for (int row = 0; row < nrows; row++) {
    for (int col = 0; col < ncols; col++) {
      idx(row, col) = columns[col * nrows + row];
    }
  }
  return idx;
}

Eigen::VectorXd toVector(double* vals, int size) {
  Eigen::VectorXd v(size);
  for (int i = 0; i < size; i++) {
    v(i) = vals[i];
  }
  return v;
}

Eigen::VectorXi toVector(int* vals, int size) {
  Eigen::VectorXi v(size);
  for (int i = 0; i < size; i++) {
    v(i) = vals[i];
  }
  return v;
}
