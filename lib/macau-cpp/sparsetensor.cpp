#include <Eigen/Dense>
#include <memory>
#include "sparsetensor.h"
#include "latentprior.h"
#include "noisemodels.h"

using namespace Eigen;

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

void MatrixData::initNoise(AdaptiveGaussianNoise* noiseModel) {
  init_noise(this, noiseModel);
}

void MatrixData::updateNoise(AdaptiveGaussianNoise* noiseModel, std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) {
  update_noise(this, noiseModel, samples);
}


inline double nCDF(double val) {return 0.5 * erfc(-val * M_SQRT1_2);}

/////  evalModel functions
void MatrixData::evalModel(ProbitNoise* noise, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var,
        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) {
  const unsigned N = Ytest.nonZeros();
  Eigen::VectorXd pred(N);
  Eigen::VectorXd test(N);
  Eigen::MatrixXd & rows = *samples[0];
  Eigen::MatrixXd & cols = *samples[1];

// #pragma omp parallel for schedule(dynamic,8) reduction(+:se, se_avg) <- dark magic :)
  for (int k = 0; k < Ytest.outerSize(); ++k) {
    int idx = Ytest.outerIndexPtr()[k];
    for (Eigen::SparseMatrix<double>::InnerIterator it(Ytest,k); it; ++it) {
     pred[idx] = nCDF(cols.col(it.col()).dot(rows.col(it.row())));
     test[idx] = it.value();

      // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
      double pred_avg;
      if (n == 0) {
        pred_avg = pred[idx];
      } else {
        double delta = pred[idx] - predictions[idx];
        pred_avg = (predictions[idx] + delta / (n + 1));
        predictions_var[idx] += delta * (pred[idx] - pred_avg);
      }
      predictions[idx++] = pred_avg;

   }
  }
  noise->auc_test_onesample = auc(pred,test);
  noise->auc_test = auc(predictions, test);
}

void MatrixData::evalModel(FixedGaussianNoise* noise, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var,
        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) {
   auto rmse = eval_rmse(Ytest, n, predictions, predictions_var, *samples[1], *samples[0], mean_value);
   noise->rmse_test = rmse.second;
   noise->rmse_test_onesample = rmse.first;
}

void MatrixData::evalModel(AdaptiveGaussianNoise* noise, const int n, Eigen::VectorXd & predictions, Eigen::VectorXd & predictions_var,
        std::vector< std::unique_ptr<Eigen::MatrixXd> > & samples) {
   auto rmse = eval_rmse(Ytest, n, predictions, predictions_var, *samples[1], *samples[0], mean_value);
   noise->rmse_test = rmse.second;
   noise->rmse_test_onesample = rmse.first;
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

template class SparseMode<3>;
template class SparseMode<4>;
template class SparseMode<5>;
template class SparseMode<6>;

template class SparseTensor<3>;
template class SparseTensor<4>;
template class SparseTensor<5>;
template class SparseTensor<6>;
