#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <memory>

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


		void setTrain(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) override;
		void setTest(int* rows, int* cols, double* values, int nnz, int nrows, int ncols) override;

    void setTrain(int* idx, int nmodes, double* values, int nnz, int* dims) override;
    void setTest(int* idx, int nmodes, double* values, int nnz, int* dims) override;

    Eigen::MatrixXd getTestData() override;
};

template<class T>
class VectorView {
	public:
		std::vector<T*> vec;
		unsigned int removed;

		VectorView(std::vector< std::unique_ptr<T> > & cvec, unsigned int cremoved) {
			for (unsigned int i = 0; i < cvec.size(); i++) {
				if (cremoved == i) {
					continue;
				}
				vec.push_back( cvec[i].get() );
			}
			removed = cremoved;
		}

		T* get(int i) {
			return vec[i];
		}

		int size() {
			return vec.size();
		}
};

//////   Tensor data   //////
class SparseMode {
  public:
    int num_modes; // number of modes
    Eigen::VectorXi row_ptr;
    Eigen::MatrixXi indices;
    Eigen::VectorXd values;
    int nnz;
    int mode;

  public:
    SparseMode() :  num_modes(0), nnz(0), mode(0) {}
    SparseMode(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size) { init(idx, vals, mode, mode_size); }
    void init(Eigen::MatrixXi &idx, Eigen::VectorXd &vals, int mode, int mode_size);
    int nonZeros() { return nnz; }
    int modeSize() { return row_ptr.size() - 1; }
};

class TensorData : public IData {
  public:
    Eigen::MatrixXi dims;
    double mean_value = .0;
    std::vector< std::unique_ptr<SparseMode> >* Y;
    SparseMode Ytest;
    int N;

  public:
    TensorData(int Nmodes) : N(Nmodes) { Y = new std::vector< std::unique_ptr<SparseMode> >(); }

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
