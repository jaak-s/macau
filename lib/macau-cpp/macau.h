#ifndef MACAU_H
#define MACAU_H

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>
#include <memory>
#include "latentprior.h"
#include "sparsetensor.h"

// try adding num_latent as template parameter to Macau
class Macau {
  public:
  int num_latent;

  //double alpha = 2.0; 
  std::unique_ptr<INoiseModel> noise;
  int nsamples = 100;
  int burnin   = 50;

//  std::unique_ptr<IData> data;
  Eigen::VectorXd predictions;
  Eigen::VectorXd predictions_var;

  double rmse_test  = .0;
  double rmse_train = .0;

  /** BPMF model */
  std::vector< std::unique_ptr<ILatentPrior> > priors;
  std::vector< std::unique_ptr<Eigen::MatrixXd> > samples;
  bool verbose = true;

  bool save_model = false;
  std::string save_prefix = "model";

  public:
    virtual void addPrior(std::unique_ptr<ILatentPrior> & prior) = 0;
    virtual void setPrecision(double p) = 0;
    virtual void setAdaptivePrecision(double sn_init, double sn_max) = 0;
    virtual void setProbit() = 0;
    virtual void setSamples(int burnin, int nsamples) = 0;
    virtual void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols) = 0;
    virtual void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) = 0;
    virtual void init() = 0;
    virtual void run() = 0;
    virtual void printStatus(int i, double elapsedi, double samples_per_sec) = 0;
    virtual void setVerbose(bool v) = 0;
    virtual double getRmseTest() = 0;
    virtual Eigen::VectorXd getPredictions() = 0;
    virtual Eigen::VectorXd getStds() = 0;
    virtual void setSaveModel(bool save) = 0;
    virtual void setSavePrefix(std::string pref) = 0;
    virtual void saveModel(int isample) = 0;
    virtual Eigen::MatrixXd getTestData() = 0;
    virtual void saveGlobalParams() = 0;
    virtual ~Macau();
};

template<class DType> 
class MacauX : public Macau {
  DType data;
  public:
    MacauX(int D) { num_latent = D; }
    MacauX() : MacauX(10) {}
    void addPrior(std::unique_ptr<ILatentPrior> & prior) override;
    void setPrecision(double p) override;
    void setAdaptivePrecision(double sn_init, double sn_max) override;
    void setProbit() override;
    void setSamples(int burnin, int nsamples) override;
    void setRelationData(int* rows, int* cols, double* values, int N, int nrows, int ncols) override;
    void setRelationDataTest(int* rows, int* cols, double* values, int N, int nrows, int ncols) override;
    void init() override;
    void run() override;
    void printStatus(int i, double elapsedi, double samples_per_sec) override;
    void setVerbose(bool v) override { verbose = v; };
    double getRmseTest() override;
    Eigen::VectorXd getPredictions() override { return predictions; };
    Eigen::VectorXd getStds() override;
    void setSaveModel(bool save) override { save_model = save; };
    void setSavePrefix(std::string pref) override { save_prefix = pref; };
    void saveModel(int isample) override;
    Eigen::MatrixXd getTestData() override { return data.getTestData(); };
    void saveGlobalParams() override;
};

Macau* make_macau(bool tensor, int num_latent);
void sparseFromIJV(Eigen::SparseMatrix<double> & X, int* rows, int* cols, double* values, int N);

#endif /* MACAU_H */
