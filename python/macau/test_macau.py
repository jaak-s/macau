import unittest
import numpy as np
import scipy.sparse
import macau

class TestMacau(unittest.TestCase):
    def test_bpmf(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = macau.make_train_test(Y, 0.5)
        results = macau.bpmf(Y, Ytest = Ytest, num_latent = 4,
                             verbose = False, burnin = 50, nsamples = 50,
                             univariate = False)
        self.assertEqual(Ytest.nnz, results.prediction.shape[0])

    def test_macau(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = macau.make_train_test(Y, 0.5)
        side1   = scipy.sparse.coo_matrix( np.random.rand(10, 2) )
        side2   = scipy.sparse.coo_matrix( np.random.rand(20, 3) )

        results = macau.bpmf(Y, Ytest = Ytest, side = [side1, side2], num_latent = 4,
                             verbose = False, burnin = 50, nsamples = 50,
                             univariate = False)
        self.assertEqual(Ytest.nnz, results.prediction.shape[0])

    def test_macau_univariate(self):
        Y = scipy.sparse.rand(10, 20, 0.2)
        Y, Ytest = macau.make_train_test(Y, 0.5)
        side1   = scipy.sparse.coo_matrix( np.random.rand(10, 2) )
        side2   = scipy.sparse.coo_matrix( np.random.rand(20, 3) )

        results = macau.bpmf(Y, Ytest = Ytest, side = [side1, side2], num_latent = 4,
                             verbose = False, burnin = 50, nsamples = 50,
                             univariate = True)
        self.assertEqual(Ytest.nnz, results.prediction.shape[0])

if __name__ == '__main__':
    unittest.main()
