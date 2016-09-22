#include <Eigen/Dense>
#include <memory>
#include "sparsetensor.h"
#include "latentprior.h"

template <class T>
void IDataDisp<T>::sample_latents(std::unique_ptr<ILatentPrior> & prior,
                                  ProbitNoise* noiseModel,
                                  std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                                  int mode,
                                  const int num_latent)
{
  prior->sample_latents(noiseModel, static_cast<T *>(this), samples, mode, num_latent);
}

template <class T>
void IDataDisp<T>::sample_latents(std::unique_ptr<ILatentPrior> & prior,
                                  AdaptiveGaussianNoise* noiseModel,
                                  std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                                  int mode,
                                  const int num_latent)
{
  prior->sample_latents(noiseModel, static_cast<T *>(this), samples, mode, num_latent);
}

template <class T>
void IDataDisp<T>::sample_latents(std::unique_ptr<ILatentPrior> & prior,
                                  FixedGaussianNoise* noiseModel,
                                  std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples,
                                  int mode,
                                  const int num_latent)
{
  prior->sample_latents(noiseModel, static_cast<T *>(this), samples, mode, num_latent);
}

template<int N>
void SparseMode<N>::init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size) {
  if (idx.rows() != vals.size()) {
    throw std::runtime_error("idx.rows() must equal vals.size()");
  }
  row_ptr.resize(mode_size + 1);
  row_ptr.setZero();
  values.resize(vals.size());
  indices.resize(idx.rows(), idx.cols() - 1);

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

template<int N>
void SparseTensor<N>::init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, Eigen::VectorXi d) {
  if (idx.rows() != vals.size()) {
    throw std::runtime_error("idx.rows() must equal vals.size()");
  }
  dims = d;
  nnz  = idx.rows();
  for (int mode = 0; mode < N; mode++) {
    sparseModes.push_back( SparseMode<N>(idx, vals, mode, dims(mode)) );
  }
}

template class SparseMode<3>;
template class SparseMode<4>;
template class SparseMode<5>;
template class SparseMode<6>;

template class SparseTensor<3>;
template class SparseTensor<4>;
template class SparseTensor<5>;
template class SparseTensor<6>;
