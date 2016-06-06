#pragma once

#include <memory>
#include <Eigen/Dense>
#include "macau.h"
#include "linop.h"

void run_macau_mpi(Macau* macau, int world_rank);
void update_prior_mpi(MacauPrior<SparseFeat> & prior, const Eigen::MatrixXd &U, int world_rank);
void sample_beta_mpi(MacauPrior<SparseFeat> &prior, const Eigen::MatrixXd &U, int world_rank);
