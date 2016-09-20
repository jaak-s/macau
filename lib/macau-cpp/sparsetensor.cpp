#include <Eigen/Dense>
#include "sparsetensor.h"

template<int N>
void SparseMode<N>::init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size) {
  row_ptr.resize(mode_size + 1);
  row_ptr.setZero();
  values.resize(vals.size());
  indices.resize(idx.rows(), idx.cols() - 1);

  auto rows = idx.col(mode);
  const int nrow = mode_size;
  const int nnz  = idx.rows();

  // compute number of non-zero entries per each element for the mode
  for (int i = 0; i < nrow; i++) {
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
