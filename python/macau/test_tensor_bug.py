import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import macau
import itertools

A = np.random.randn(15, 2)
B = np.random.randn(20, 2)
C = np.random.randn(1, 2)

idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])
Ytrain, Ytest = macau.make_train_test_df(df, 0.2)

results = macau.bpmf(Y = Ytrain, Ytest = Ytest, num_latent = 4,
                     verbose = True, burnin = 20, nsamples = 2,
                     univariate = False, precision = 50)

Ytrain_sp = scipy.sparse.coo_matrix( (Ytrain.value, (Ytrain.A, Ytrain.B) ) )
Ytest_sp  = scipy.sparse.coo_matrix( (Ytest.value,  (Ytest.A, Ytest.B) ) )

results = macau.bpmf(Y = Ytrain_sp, Ytest = Ytest_sp, num_latent = 4,
                     verbose = True, burnin = 20, nsamples = 2,
                     univariate = False, precision = 50)
