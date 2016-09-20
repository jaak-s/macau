#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

template<int N, typename T>
class SparseTensor {
  public:
    int dim(unsigned int m);
    unsigned int nonZeros();

}
