import unittest
import numpy as np
import pandas as pd
import scipy.sparse
import macau
import itertools

class TestMacau(unittest.TestCase):
    def test_bpmf(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = macau.make_train_test(Y, 0.5)
        results = macau.bpmf(Y, Ytest = Ytest, num_latent = 4,
                             verbose = False, burnin = 50, nsamples = 50,
                             univariate = False)
        self.assertEqual(Ytest.nnz, results.prediction.shape[0])

    def test_bpmf_numerictest(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xt = 0.3
        macau.bpmf(X, Xt, num_latent = 10, burnin=10, nsamples=15, verbose = False)

    def test_macau(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = macau.make_train_test(Y, 0.5)
        side1   = scipy.sparse.coo_matrix( np.random.rand(10, 2) )
        side2   = scipy.sparse.coo_matrix( np.random.rand(20, 3) )

        results = macau.bpmf(Y, Ytest = Ytest, side = [side1, side2], num_latent = 4,
                             verbose = False, burnin = 50, nsamples = 50,
                             univariate = False)
        self.assertEqual(Ytest.nnz, results.prediction.shape[0])

    def test_macau_side_bin(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xt = scipy.sparse.rand(15, 10, 0.1)
        F = scipy.sparse.rand(15, 2, 0.5)
        F.data[:] = 1
        macau.macau(X, Xt, side=[F, None], num_latent = 5, burnin=10, nsamples=5, verbose = False)


    def test_macau_univariate(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = macau.make_train_test(Y, 0.5)
        side1   = scipy.sparse.coo_matrix( np.random.rand(10, 2) )
        side2   = scipy.sparse.coo_matrix( np.random.rand(20, 3) )

        results = macau.bpmf(Y, Ytest = Ytest, side = [side1, side2], num_latent = 4,
                             verbose = False, burnin = 50, nsamples = 50,
                             univariate = True)
        self.assertEqual(Ytest.nnz, results.prediction.shape[0])

    def test_too_many_sides(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        with self.assertRaises(ValueError):
            macau.macau(Y, verbose = False, side = [None, None, None])

    def test_bpmf_emptytest(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        macau.bpmf(X, Ytest = 0, num_latent = 10, burnin=10, nsamples=15, verbose=False)
        
    def test_bpmf_emptytest_probit(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        X.data = X.data > 0.5
        macau.bpmf(X, Ytest = 0, num_latent = 10, burnin=10, nsamples=15, precision="probit", verbose=False)
        macau.bpmf(X, Ytest = None, num_latent = 10, burnin=10, nsamples=15, precision="probit", verbose=False)

    def test_make_train_test(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xtr, Xte = macau.make_train_test(X, 0.5)
        self.assertEqual(X.nnz, Xtr.nnz + Xte.nnz)
        diff = np.linalg.norm( (X - Xtr - Xte).todense() )
        self.assertEqual(diff, 0.0)

    def test_bpmf_tensor(self):
        np.random.seed(1234)
        Y = pd.DataFrame({
            "A": np.random.randint(0, 5, 7),
            "B": np.random.randint(0, 4, 7),
            "C": np.random.randint(0, 3, 7),
            "value": np.random.randn(7)
        })
        Ytest = pd.DataFrame({
            "A": np.random.randint(0, 5, 5),
            "B": np.random.randint(0, 4, 5),
            "C": np.random.randint(0, 3, 5),
            "value": np.random.randn(5)
        })
        results = macau.bpmf(Y, Ytest = Ytest, num_latent = 4,
                             verbose = False, burnin = 50, nsamples = 50,
                             univariate = False)

    def test_bpmf_tensor2(self):
        A = np.random.randn(15, 2)
        B = np.random.randn(20, 2)
        C = np.random.randn(3, 2)

        idx = list( itertools.product(np.arange(A.shape[0]), np.arange(B.shape[0]), np.arange(C.shape[0])) )
        df  = pd.DataFrame( np.asarray(idx), columns=["A", "B", "C"])
        df["value"] = np.array([ np.sum(A[i[0], :] * B[i[1], :] * C[i[2], :]) for i in idx ])

        results = macau.bpmf(Y = df, Ytest = 0.2, num_latent = 4,
                             verbose = True, burnin = 20, nsamples = 20,
                             univariate = False, precision = 50)


if __name__ == '__main__':
    unittest.main()
