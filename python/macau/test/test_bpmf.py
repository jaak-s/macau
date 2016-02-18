import unittest
import macau
import scipy.sparse

class TestBPMF(unittest.TestCase):
    def test_bpmf(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        macau.bpmf(X, num_latent = 10, burnin=10, nsamples=15)
    def test_bpmf2(self):
        X = scipy.sparse.rand(15, 10, 0.2)
        Xt = scipy.sparse.rand(15, 10, 0.1)
        macau.bpmf(X, Xt, num_latent = 10, burnin=10, nsamples=15)

if __name__ == '__main__':
    unittest.main()
