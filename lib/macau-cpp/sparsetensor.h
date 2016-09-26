#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

#include "latentprior.h"
#include "noisemodels.h"
#include "bpmfutils.h"


// forward declarations
class ILatentPrior;
class ProbitNoise;
class AdaptiveGaussianNoise;
class FixedGaussianNoise;

class IData {
  public:
    virtual void setTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) = 0;
    virtual Eigen::VectorXi & getDims() = 0;
    virtual int getTestNonzeros() = 0;
    virtual Eigen::MatrixXd getTestData() = 0;
    virtual double getMeanValue() = 0;
    virtual ~IData() {};
};

//////   Matrix data    /////
class MatrixData : public IData {
  public:
    Eigen::SparseMatrix<double> Y, Yt, Ytest;
    double mean_value = .0; 
    Eigen::VectorXi dims;
    
    MatrixData() {}

    Eigen::VectorXi & getDims() { return dims; }
    int getTestNonzeros() { return Ytest.nonZeros(); }
    double getMeanValue() override { return mean_value; }

    void setTrain(int* rows, int* cols, double* values, int N, int nrows, int ncols) {
      Y.resize(nrows, ncols);
      sparseFromIJV(Y, rows, cols, values, N);
      Yt = Y.transpose();
      mean_value = Y.sum() / Y.nonZeros();
      dims.resize(2);
      dims << Y.rows(), Y.cols();
    }

    void setTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) override {
      Ytest.resize(nrows, ncols);
      sparseFromIJV(Ytest, rows, cols, values, N);
    }

    Eigen::MatrixXd getTestData() override;
};

//////   Tensor data   //////
template<int N>
class SparseMode {
  public:
    Eigen::VectorXi row_ptr;
    Eigen::Matrix< int, Eigen::Dynamic, N-1 > indices;
    Eigen::VectorXd values;
    int nnz;

  public:
    SparseMode(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size) { init(idx, vals, mode, mode_size); }
    void init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size);
};

template<int N>
class TensorData {
  public:
    Eigen::Matrix< int, N, 1 > dims;
    int nnz;
    int nonZeros() { return nnz; };
    std::vector< SparseMode<N> > Y;

  public:
    TensorData(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi dims) { init(idx, vals, dims); }
    void init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi d);
};
