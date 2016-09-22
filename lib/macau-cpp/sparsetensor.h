#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

#include "latentprior.h"
#include "noisemodels.h"

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
