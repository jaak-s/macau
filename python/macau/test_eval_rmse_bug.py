import numpy as np
import pandas as pd
import scipy.sparse
import macau

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

