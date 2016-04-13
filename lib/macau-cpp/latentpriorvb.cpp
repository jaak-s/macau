#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "latentpriorvb.h"

using namespace std; 
using namespace Eigen;

void BPMFPriorVB::init(const int num_latent) {
}

void BPMFPriorVB::update_latents(
    Eigen::MatrixXd &Umean,
    Eigen::MatrixXd &Uvar,
    const Eigen::SparseMatrix<double> &Y,
    const double mean_value,
    const Eigen::MatrixXd &Vmean,
    const Eigen::MatrixXd &Vvar,
    const double alpha) {
}

void BPMFPriorVB::update_prior(const Eigen::MatrixXd &Umean, const Eigen::MatrixXd &Uvar) {
}
