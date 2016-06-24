#include <mpi.h>
#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <iostream>
#include <fstream>

#include <fstream>
#include <string>
#include <algorithm>
#include <random>
#include <chrono>
#include <memory>
#include <cmath>
#include <stdlib.h>

#include <getopt.h>

#include <unsupported/Eigen/SparseExtra>

#include "linop.h"
#include "macau_mpi.h"
#include "macau.h"

extern "C" {
  #include "dsparse.h"
}

using namespace Eigen;
using namespace std;

void usage() {
   printf("Usage:\n");
   printf("  macau_mpi --train <train_file> --row-features <feature-file> [options]\n");
   printf("Optional:\n");
   printf("  --test    test_file  test data (for computing RMSE)\n");
   printf("  --burnin        200  number of samples to discard\n");
   printf("  --nsamples      800  number of samples to collect\n");
   printf("  --num-latent     96  number of latent dimensions\n");
   printf("  --precision     5.0  precision of observations\n");
   printf("  --lambda-beta  10.0  initial value of lambda beta\n");
   printf("  --tol          1e-6  tolerance for CG\n");
   printf("  --output    results  prefix for result files\n");
}

bool file_exists(const char *fileName)
{
   std::ifstream infile(fileName);
   return infile.good();
}

int get_num_omp_threads() {
   int nt = 0;
#pragma omp parallel
   {
#pragma omp single
      {
         nt = omp_get_num_threads();
      }
   }
   return nt;
}

void die(std::string message, int world_rank) {
   if (world_rank == 0) {
      std::cout << message;
   }
   MPI_Finalize();
   exit(1);
}

std::unique_ptr<SparseFeat> load_bcsr(const char* filename) {
   SparseBinaryMatrix* A = read_sbm(filename);
   SparseFeat* sf = new SparseFeat(A->nrow, A->ncol, A->nnz, A->rows, A->cols);
   free_sbm(A);
   std::unique_ptr<SparseFeat> sf_ptr(sf);
   return sf_ptr;
}

// var for MPI
int* rhs_for_rank = NULL;
double* rec     = NULL;
int* sendcounts = NULL;
int* displs     = NULL;

int main(int argc, char** argv) {
   // Initialize the MPI environment
   MPI_Init(NULL, NULL);
   // Get the number of processes
   int world_size, world_rank;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size);
   MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

   // Get the name of the processor
   char processor_name[MPI_MAX_PROCESSOR_NAME];
   int name_len;
   MPI_Get_processor_name(processor_name, &name_len);

   char* fname_train         = NULL;
   char* fname_test          = NULL;
   char* fname_row_features  = NULL;
   std::string output_prefix = std::string("result");
   double precision   = 5.0;
   double lambda_beta = 10.0;
   double tol         = 1e-6;
   int burnin     = 200;
   int nsamples   = 800;
   int num_latent = 96;

   // reading command line arguments
   while (1) {
      static struct option long_options[] =
      {
         {"train",      required_argument, 0, 't'},
         {"test",       required_argument, 0, 'e'},
         {"row-features", required_argument, 0, 'r'},
         {"precision",  required_argument, 0, 'p'},
         {"burnin",     required_argument, 0, 'b'},
         {"nsamples",   required_argument, 0, 'n'},
         {"output",     required_argument, 0, 'o'},
         {"num-latent", required_argument, 0, 'l'},
         {"lambda-beta",required_argument, 0, 'a'},
         {"tol",        required_argument, 0, 'c'},
         {0, 0, 0, 0}
      };
      int option_index = 0;
      int c = getopt_long(argc, argv, "t:e:r:p:b:n:o:a:c:", long_options, &option_index);
      if (c == -1)
         break;

      switch (c) {
         case 'a': lambda_beta   = strtod(optarg, NULL); break;
         case 'b': burnin        = strtol(optarg, NULL, 10); break;
         case 'c': tol           = atof(optarg); break;
         case 'e': fname_test    = optarg; break;
         case 'l': num_latent    = strtol(optarg, NULL, 10); break;
         case 'n': nsamples      = strtol(optarg, NULL, 10); break;
         case 'o': output_prefix = std::string(optarg); break;
         case 'p': precision     = strtod(optarg, NULL); break;
         case 'r': fname_row_features = optarg; break;
         case 't': fname_train = optarg; break;
         case '?':
         default:
           if (world_rank == 0)
              usage();
           MPI_Finalize();
           exit(1);
      }
   }
   if (fname_train == NULL || fname_row_features == NULL) {
      if (world_rank == 0) {
         printf("[ERROR]\nMissing parameters '--matrix' or '--row-features'.\n");
         usage();
      }
      MPI_Finalize();
      exit(1);
   }
   if (world_rank == 0) {
      printf("Train data:    '%s'\n", fname_train);
      printf("Test data:     '%s'\n", fname_test==NULL ?"" :fname_test);
      printf("Row features:  '%s'\n", fname_row_features);
      printf("Output prefix: '%s'\n", output_prefix.c_str());
      printf("Burn-in:       %d\n", burnin);
      printf("Samples:       %d\n", nsamples);
      printf("Num-latents:   %d\n", num_latent);
      printf("Precision:     %.1f\n", precision);
      printf("Lambda-beta:   %.1f\n", lambda_beta);
      printf("tol:           %.1e\n", tol);
   }
   if ( ! file_exists(fname_train) ) {
      die(std::string("[ERROR]\nTrain data file '") + fname_train + "' not found.\n", world_rank);
   }
   if ( ! file_exists(fname_row_features) ) {
      die(std::string("[ERROR]\nRow feature file '") + fname_row_features + "' not found.\n", world_rank);
   }
   if ( (fname_test != NULL) && ! file_exists(fname_test) ) {
      die(std::string("[ERROR]\nTest data file '") + fname_test + "' not found.\n", world_rank);
   }

   int n_omp_threads = get_num_omp_threads();
   rhs_for_rank = new int[world_size];
   split_work_mpi(num_latent, world_size, rhs_for_rank);

   // Print off a hello world message
   printf("Processor %s, rank %d"
          " out of %d processors using %d OpenMP threads for %d RHS.\n",
          processor_name, world_rank, world_size, n_omp_threads, rhs_for_rank[world_rank]);

   // Step 1. Loading data
   //std::unique_ptr<SparseFeat> row_features = load_bcsr(fname_row_features);
   auto row_features = load_bcsr(fname_row_features);
   if (world_rank == 0) {
      printf("Row features:   [%d x %d].\n", row_features->rows(), row_features->cols());
   }
   sendcounts = new int[world_size];
   displs     = new int[world_size];
   int sum = 0;
   for (int n = 0; n < world_size; n++) {
      sendcounts[n] = rhs_for_rank[n] * row_features->cols();
      displs[n]     = sum;
      sum          += sendcounts[n];
   }
   rec = new double[sendcounts[world_rank]];

   SparseDoubleMatrix* Y     = NULL;
   SparseDoubleMatrix* Ytest = NULL;

   Macau* macau = new Macau(num_latent);
   macau->setPrecision(precision);
   macau->setSamples(burnin, nsamples);
   macau->setVerbose(true);

   // 1) row prior with side information
   int nfeat    = row_features->rows();
   auto prior_u = new MacauPrior<SparseFeat>(num_latent, row_features, false);
   prior_u->setLambdaBeta(lambda_beta);
   prior_u->setTol(tol);
   auto prior_v = new BPMFPrior(num_latent);

   // 2) activity data (read_sdm)
   Y = read_sdm(fname_train);

   if (nfeat != Y->nrow) {
      die(std::string("[ERROR]\nNumber of rows (" +
                      std::to_string(Y->nrow) +
                      ") in train must be equal to number of rows in row-features (" +
                      std::to_string(row_features->rows()) +
                      ")."),
          world_rank);
   }

   // 3) create Macau object
   macau->setRelationData(Y->rows, Y->cols, Y->vals, Y->nnz, Y->nrow, Y->ncol);

   // test data
   if (fname_test != NULL) {
      Ytest = read_sdm(fname_test);
      if (Ytest->nrow != Y->nrow || Ytest->ncol != Y->ncol) {
         die(std::string("[ERROR]\nSize of train (") +
                         std::to_string(Y->nrow) + " x " +
                         std::to_string(Y->ncol) + ") must be equal to size of test (" +
                         std::to_string(Ytest->nrow) + " x " +
                         std::to_string(Ytest->ncol) + ").",
             world_rank);
      }
      macau->setRelationDataTest(Ytest->rows, Ytest->cols, Ytest->vals, Ytest->nnz, Ytest->nrow, Ytest->ncol);
   }
   std::unique_ptr<ILatentPrior> u_ptr(prior_u);
   std::unique_ptr<ILatentPrior> v_ptr(prior_v);
   macau->addPrior( u_ptr );
   macau->addPrior( v_ptr );

   if (world_rank == 0) {
      printf("Training data:  %ld [%d x %d]\n", Y->nnz, Y->nrow, Y->ncol);
      if (Ytest != NULL) {
         printf("Test data:      %ld [%d x %d]\n", Ytest->nnz, Ytest->nrow, Ytest->ncol);
      } else {
         printf("Test data:      --\n");
      }
   }

   run_macau_mpi(macau, world_rank);

   // save results
   if (world_rank == 0) {
      VectorXd yhat_raw     = macau->getPredictions();
      VectorXd yhat_sd_raw  = macau->getStds();
      MatrixXd testdata_raw = macau->getTestData();

      std::string fname_pred = output_prefix + "-predictions.csv";
      std::ofstream predfile;
      predfile.open(fname_pred);
      predfile << "row,col,y,y_pred,y_pred_std\n";
      for (int i = 0; i < yhat_raw.size(); i++) {
         predfile << to_string( (int)testdata_raw(i,0) );
         predfile << "," << to_string( (int)testdata_raw(i,1) );
         predfile << "," << to_string( testdata_raw(i,2) );
         predfile << "," << to_string( yhat_raw(i) );
         predfile << "," << to_string( yhat_sd_raw(i) );
         predfile << "\n";
      }
      predfile.close();
      printf("Saved predictions into '%s'.\n", fname_pred.c_str());
   }

   // Finalize the MPI environment.
   delete macau;
   MPI_Finalize();
   return 0;
}

void run_macau_mpi(
      Macau* macau,
      int world_rank)
{
   /* adapted from Macau.run() */
   macau->init();
   if (world_rank == 0 && macau->verbose) {
      std::cout << "Sampling" << std::endl;
   }

   const int num_rows = macau->Y.rows();
   const int num_cols = macau->Y.cols();
   macau->predictions     = VectorXd::Zero( macau->Ytest.nonZeros() );
   macau->predictions_var = VectorXd::Zero( macau->Ytest.nonZeros() );

   auto start = tick();
   for (int i = 0; i < macau->burnin + macau->nsamples; i++) {
      if (world_rank == 0 && macau->verbose && i == macau->burnin) {
         printf(" ====== Burn-in complete, averaging samples ====== \n");
      }
      auto starti = tick();

      if (world_rank == 0) {
         // sample latent vectors
         macau->priors[0]->sample_latents(*macau->samples[0], macau->Yt, macau->mean_rating, *macau->samples[1], macau->alpha, macau->num_latent);
         macau->priors[1]->sample_latents(*macau->samples[1], macau->Y,  macau->mean_rating, *macau->samples[0], macau->alpha, macau->num_latent);
      }

      // Sample hyperparams
      update_prior_mpi( *(MacauPrior<SparseFeat>*) macau->priors[0].get(), *macau->samples[0], world_rank);
      if (world_rank == 0) {
         macau->priors[1]->update_prior(*macau->samples[1]);

         auto eval = eval_rmse(macau->Ytest, (i < macau->burnin) ? 0 : (i - macau->burnin), macau->predictions, macau->predictions_var,
                               *macau->samples[1], *macau->samples[0], macau->mean_rating);

         auto endi = tick();
         auto elapsed = endi - start;
         double samples_per_sec = (i + 1) * (num_rows + num_cols) / elapsed;
         double elapsedi = endi - starti;

         if (macau->verbose) {
           macau->printStatus(i, eval.first, eval.second, elapsedi, samples_per_sec);
         }
         macau->rmse_test = eval.second;
      }
   }
}

void update_prior_mpi(MacauPrior<SparseFeat> &prior, const Eigen::MatrixXd &U, int world_rank) {
   if (world_rank == 0) {
      // residual (Uhat is later overwritten):
      prior.Uhat.noalias() = U - prior.Uhat;
      MatrixXd BBt = A_mul_At_combo(prior.beta);
      // sampling Gaussian
      std::tie(prior.mu, prior.Lambda) = CondNormalWishart(prior.Uhat, prior.mu0, prior.b0, prior.WI + prior.lambda_beta * BBt, prior.df + prior.beta.cols());
   }
   sample_beta_mpi(prior, U, world_rank);
   if (world_rank == 0) { 
      compute_uhat(prior.Uhat, *prior.F, prior.beta);
      prior.lambda_beta = sample_lambda_beta(prior.beta, prior.Lambda, prior.lambda_beta_nu0, prior.lambda_beta_mu0);
   }
}

void sample_beta_mpi(MacauPrior<SparseFeat> &prior, const Eigen::MatrixXd &U, int world_rank) {
   const int num_latent = prior.beta.rows();
   const int num_feat = prior.beta.cols();
   MatrixXd Ft_y(0,0);

   if (world_rank == 0) {
      // Ft_y = (U .- mu + Normal(0, Lambda^-1)) * F + sqrt(lambda_beta) * Normal(0, Lambda^-1)
      // Ft_y is [ D x F ] matrix
      MatrixXd tmp = (U + MvNormal_prec_omp(prior.Lambda, U.cols())).colwise() - prior.mu;
      Ft_y = A_mul_B(tmp, *prior.F);
      MatrixXd tmp2 = MvNormal_prec_omp(prior.Lambda, num_feat);

      #pragma omp parallel for schedule(static)
      for (int f = 0; f < num_feat; f++) {
         for (int d = 0; d < num_latent; d++) {
            Ft_y(d, f) += sqrt(prior.lambda_beta) * tmp2(d, f);
         }
      }
      Ft_y.transposeInPlace();
   }

   MPI_Bcast(& prior.lambda_beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   MPI_Bcast(& prior.tol,         1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   // sending Ft_y
   MPI_Scatterv(Ft_y.data(), sendcounts, displs, MPI_DOUBLE, rec, sendcounts[world_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
   int nrhs = rhs_for_rank[world_rank];
   MatrixXd RHS(nrhs, num_feat), result(nrhs, num_feat);

#pragma omp parallel for schedule(static)
   for (int f = 0; f < num_feat; f++) {
      for (int d = 0; d < nrhs; d++) {
         RHS(d, f) = rec[f + d * num_feat];
      }
   }
   // solving
   solve_blockcg(result, *prior.F, prior.lambda_beta, RHS, prior.tol, 32, 8);
   result.transposeInPlace();
   MPI_Gatherv(result.data(), nrhs*num_feat, MPI_DOUBLE, Ft_y.data(), sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   if (world_rank == 0) {
      //prior.beta = Ft_y.transpose();
#pragma omp parallel for schedule(static)
      for (int f = 0; f < num_feat; f++) {
         for (int d = 0; d < num_latent; d++) {
            prior.beta(d, f) = Ft_y(f, d);
         }
      }
   }
}
