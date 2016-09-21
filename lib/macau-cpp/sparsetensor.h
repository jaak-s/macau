#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

template<int N>
class SparseMode {
  public:
    Eigen::VectorXi row_ptr;
    Eigen::Matrix< int, Eigen::Dynamic, N-1 > indices;
    Eigen::VectorXd values;
    long nnz;

  public:
    SparseMode(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size) { init(idx, vals, mode, mode_size); }
    void init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size);
};

template<int N>
class SparseTensor {
  public:
    Eigen::Matrix< int, N, 1 > dims;
    long nnz;
    long nonZeros() { return nnz; };
    std::vector< SparseMode<N> > sparseModes;

  public:
    SparseTensor(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi dims) { init(idx, vals, dims); }
    void init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi d);

};
