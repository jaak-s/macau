#include <Eigen/Dense>
#include <memory>
#include <iostream>
#include "sparsetensor.h"
#include "latentprior.h"
#include "noisemodels.h"

using namespace Eigen;

template<int N>
void SparseMode<N>::init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int m, int mode_size) {
  if (idx.rows() != vals.size()) {
    throw std::runtime_error("idx.rows() must equal vals.size()");
  }
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


////////  TensorData  ///////

template<int N>
void TensorData<N>::setTrain(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi &d) {
  if (idx.rows() != vals.size()) {
    throw std::runtime_error("setTrain(): idx.rows() must equal vals.size()");
  }
  dims = d;
  for (int mode = 0; mode < N; mode++) {
    Y.push_back( SparseMode<N>(idx, vals, mode, dims(mode)) );
  }
}

template<int N>
void TensorData<N>::setTrain(int** columns, int nmodes, double* values, int nnz, int* dims) {
  auto idx  = toMatrix(columns, nnz, nmodes);
  auto vals = toVector(values, nnz);
  auto d    = toVector(dims, nmodes);
  setTrain(idx, vals, d);
}
    
template<int N>
void TensorData<N>::setTest(int** columns, int nmodes, double* values, int nnz, int* dims) {
  auto idx  = toMatrix(columns, nnz, nmodes);
  auto vals = toVector(values, nnz);
  auto d    = toVector(dims, nmodes);
  setTest(idx, vals, d);
}

template<int N>
void TensorData<N>::setTest(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi &d) {
  if (idx.rows() != vals.size()) {
    throw std::runtime_error("setTest(): idx.rows() must equal vals.size()");
  }
  if ((d - dims).norm() != 0) {
    throw std::runtime_error("setTest(): train and test Tensor sizes are not equal.");
  }
  Ytest = SparseMode<N>(idx, vals, 0, d(0));
}

template<int N>
void TensorData<N>::setTrain(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) {
  auto idx  = toMatrix(rows, cols, nnz);
  auto vals = toVector(values, nnz);

  VectorXi d(2);
  d << nrows, ncols;

  setTrain(idx, vals, d);
}

template<int N>
void TensorData<N>::setTest(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) {
  auto idx  = toMatrix(rows, cols, nnz);
  auto vals = toVector(values, nnz);

  VectorXi d(2);
  d << nrows, ncols;

  setTest(idx, vals, d);
}

template<int N>
Eigen::MatrixXd TensorData<N>::getTestData() {
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
  int** ptr = new int*[2];
  ptr[0] = col1;
  ptr[1] = col2;
  auto idx = toMatrix(ptr, nrows, 2);
  delete ptr;
  return idx;
}

Eigen::MatrixXi toMatrix(int** columns, int nrows, int ncols) {
  Eigen::MatrixXi idx(nrows, ncols);
  for (int row = 0; row < nrows; row++) {
    for (int col = 0; col < ncols; col++) {
      idx(row, col) = columns[col][row];
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

template class SparseMode<3>;
template class SparseMode<4>;
template class SparseMode<5>;
template class SparseMode<6>;

template class TensorData<3>;
template class TensorData<4>;
template class TensorData<5>;
template class TensorData<6>;
