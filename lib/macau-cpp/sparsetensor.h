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
    virtual void sample_latents(std::unique_ptr<ILatentPrior> & prior,
                                ProbitNoise* noiseModel,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                                int mode,
                                const int num_latent) = 0;
    virtual void sample_latents(std::unique_ptr<ILatentPrior> & prior,
                                AdaptiveGaussianNoise* noiseModel,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                                int mode,
                                const int num_latent) = 0;
    virtual void sample_latents(std::unique_ptr<ILatentPrior> & prior,
                                FixedGaussianNoise* noiseModel,
                                std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                                int mode,
                                const int num_latent) = 0;
    virtual void initNoise(ProbitNoise* noiseModel) = 0;
    virtual void initNoise(AdaptiveGaussianNoise* noiseModel) = 0;
    virtual void initNoise(FixedGaussianNoise* noiseModel) = 0;
    virtual void setTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) = 0;
    virtual Eigen::VectorXi & getDims() = 0;
    virtual long getTestNonzeros() = 0;
    virtual ~IData() {};
};

template<class T>
class IDataDisp : public IData {
    void sample_latents(std::unique_ptr<ILatentPrior> & prior,
                        ProbitNoise* noiseModel,
                        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                        int mode,
                        const int num_latent) override;
    void sample_latents(std::unique_ptr<ILatentPrior> & prior,
                        AdaptiveGaussianNoise* noiseModel,
                        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                        int mode,
                        const int num_latent) override;
    void sample_latents(std::unique_ptr<ILatentPrior> & prior,
                        FixedGaussianNoise* noiseModel,
                        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                        int mode,
                        const int num_latent) override;
};

//////   Matrix data    /////
class MatrixData : public IDataDisp<MatrixData> {
  public:
    Eigen::SparseMatrix<double> Y, Yt, Ytest;
    double mean_value = .0; 
    Eigen::VectorXi dims;
    
    MatrixData() {}

    Eigen::VectorXi & getDims() { return dims; }
    long getTestNonzeros() { return Ytest.nonZeros(); }

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

    virtual void initNoise(ProbitNoise* noiseModel)         { /* no init needed */ };
    virtual void initNoise(FixedGaussianNoise* noiseModel)  { /* no init needed */ };
    virtual void initNoise(AdaptiveGaussianNoise* noiseModel) override;
};

//////   Tensor data   //////
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
