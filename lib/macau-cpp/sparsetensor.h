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
    virtual void setTrain(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) = 0;
    virtual void setTest(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) = 0;
    virtual void setTrain(int* idx, int nmodes, double* values, int nnz, int* dims) = 0;
    virtual void setTest(int* idx, int nmodes, double* values, int nnz, int* dims) = 0;
    virtual Eigen::VectorXi getDims() = 0;
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

    Eigen::VectorXi getDims() { return dims; }
    int getTestNonzeros() { return Ytest.nonZeros(); }
    double getMeanValue() override { return mean_value; }

    void setTrain(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) override {
      Y.resize(nrows, ncols);
      sparseFromIJV(Y, rows, cols, values, nnz);
      Yt = Y.transpose();
      mean_value = Y.sum() / Y.nonZeros();
      dims.resize(2);
      dims << Y.rows(), Y.cols();
    }

    void setTest(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) override {
      Ytest.resize(nrows, ncols);
      sparseFromIJV(Ytest, rows, cols, values, nnz);
    }

    void setTrain(int* idx, int nmodes, double* values, int nnz, int* dims) override;
    void setTest(int* idx, int nmodes, double* values, int nnz, int* dims) override;

    Eigen::MatrixXd getTestData() override;
};

//////   Tensor data   //////
class SparseMode {
  public:
    int N; // 
    Eigen::VectorXi row_ptr;
    Eigen::MatrixXi indices;
    Eigen::VectorXd values;
    int nnz;
    int mode;

  public:
    SparseMode() {}
    SparseMode(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size) { init(idx, vals, mode, mode_size); }
    void init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size);
    int nonZeros() { return nnz; }
    int modeSize() { return row_ptr.size() - 1; }
};

class TensorData : public IData {
  public:
    Eigen::MatrixXi dims;
    double mean_value;
    std::vector< SparseMode > Y;
    SparseMode Ytest;
    int N;

  public:
    TensorData(int Nmodes) : N(Nmodes) { }

    void setTrain(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi &d);
    void setTest(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi &d);
    void setTrain(int* idx, int nmodes, double* values, int nnz, int* dims) override;
    void setTest(int* idx, int nmodes, double* values, int nnz, int* dims) override;

    void setTrain(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) override;
    void setTest(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) override;
    Eigen::VectorXi getDims() override { return dims; };
    int getTestNonzeros() override { return Ytest.nonZeros(); };
    Eigen::MatrixXd getTestData() override;
    double getMeanValue() override { return mean_value; };
};

Eigen::MatrixXi toMatrix(int* columns, int nrows, int ncols);
Eigen::MatrixXi toMatrix(int* col1, int* col2, int nrows);
Eigen::VectorXd toVector(double* vals, int size);
Eigen::VectorXi toVector(int* vals, int size);
