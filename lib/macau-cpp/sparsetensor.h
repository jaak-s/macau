#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

template<int N>
class SparseMode {
  public:
    Eigen::VectorXi row_ptr;
    Eigen::Matrix< int, Eigen::Dynamic, N-1 > indices;
    Eigen::VectorXd values;

  public:
    SparseMode(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size) { init(idx, vals, mode, mode_size); }
    void init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size);
};

template<int N>
class SparseTensor {
  public:
    int dim(unsigned int m);
    unsigned int nonZeros();

};
