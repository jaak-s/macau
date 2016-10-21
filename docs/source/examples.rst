.. role:: python(code)
   :language: python

Examples
===========
In these examples we use ChEMBL dataset for compound-proteins activities (IC50). The IC50 values and ECFP fingerprints can be downloaded from these two urls:

.. code-block:: bash

   wget http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm
   wget http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm

Matrix Factorization with Side Information
-------------------------------------------

In this example we use MCMC (Gibbs) sampling to perform factorization of the `compound x protein` IC50 matrix by using side information on the compounds.

.. code-block:: python

   import macau
   import scipy.io

   ## loading data
   ic50 = scipy.io.mmread("chembl-IC50-346targets.mm")
   ecfp = scipy.io.mmread("chembl-IC50-compound-feat.mm")

   ## running factorization (Macau)
   result = macau.macau(Y = ic50,
                        Ytest      = 0.2,
                        side       = [ecfp, None],
                        num_latent = 32,
                        precision  = 5.0,
                        burnin     = 400,
                        nsamples   = 1600)

Input matrix for :python:`Y` is a sparse scipy matrix (either coo_matrix, csr_matrix or csc_matrix).

In this example, we have assigned 20% of the IC50 data to the test set by setting :python:`Ytest = 0.2`.
If you want to use a predefined test data, set :python:`Ytest = my_test_matrix`, where the matrix is a sparse matrix of the same size as :python:`Y`.
Here we have used burn-in of 400 samples for the Gibbs sampler and then collected 1600 samples from the model.
This is usually sufficient. For quick runs smaller numbers can be used, like :python:`burnin = 100, nsamples = 500`.

The parameter :python:`side = [ecfp, None]` sets the side information for rows and columns, respectively.
In this example we only use side information for the compounds (rows of the matrix).

The :python:`precision = 5.0` specifies the precision of the IC50 observations, i.e., 1 / variance.

When the run has completed you can check the :python:`result` object and its :python:`prediction` field, which is a Pandas DataFrame.

.. code-block:: python

   >>> result
   Matrix factorization results
   Test RMSE:        0.6393
   Matrix size:      [15073 x 346]
   Number of train:  47424
   Number of test:   11856
   To see predictions on test set see '.prediction' field.

   >>> result.prediction
           col   row    y     y_pred      y_pred_std
   0        0   2233  5.7721  5.750984    1.177526
   1        0   2354  5.0947  5.379610    0.857858
   ...

Univariate sampler
~~~~~~~~~~~~~~~~~~
Macau also includes an option to use a **very fast** univariate sampler, i.e., instead of sampling blocks of variables jointly it samples each individually.
An example:

.. code-block:: python
   :emphasize-lines: 14

   import macau
   import scipy.io

   ## loading data
   ic50 = scipy.io.mmread("chembl-IC50-346targets.mm")
   ecfp = scipy.io.mmread("chembl-IC50-compound-feat.mm")

   ## running factorization (Macau)
   result = macau.macau(Y = ic50,
                        Ytest      = 0.2,
                        side       = [ecfp, None],
                        num_latent = 32,
                        precision  = 5.0,
                        univariate = True,
                        burnin     = 500,
                        nsamples   = 3500)


When using it we recommend using larger values for :python:`burnin` and :python:`nsamples`, because the univariate sampler mixes slower than the blocked sampler.

Adaptive noise
~~~~~~~~~~~~~~
In the previous examples we fixed the observation noise by specifying :python:`precision = 5.0`.
Instead we can also allow the model to automatically determine the precision of the noise by setting :python:`precision = "adaptive"`.

.. code-block:: python
   :emphasize-lines: 13

   import macau
   import scipy.io

   ## loading data
   ic50 = scipy.io.mmread("chembl-IC50-346targets.mm")
   ecfp = scipy.io.mmread("chembl-IC50-compound-feat.mm")

   ## running factorization (Macau)
   result = macau.macau(Y = ic50,
                        Ytest      = 0.2,
                        side       = [ecfp, None],
                        num_latent = 32,
                        precision  = "adaptive",
                        univariate = True,
                        burnin     = 500,
                        nsamples   = 3500)

In the case of adaptive noise the model updates (samples) the precision parameter in every iteration, which is then also shown in the output.
Additionally, there is a parameter :python:`sn_max` that sets the maximum allowed signal-to-noise ratio.
This means that if the updated precision would imply a higher signal-to-noise ratio than :python:`sn_max`, then the precision value is set to :python:`(sn_max + 1.0) / Yvar` where Yvar is the variance of the training dataset :python:`Y`.

Matrix Factorization without Side Information
----------------------------------------------
To run matrix factorization without side information you can just drop the :python:`side` parameter.

.. code-block:: python

   result = macau.macau(Y = ic50,
                        Ytest      = 0.2,
                        num_latent = 32,
                        precision  = 5.0,
                        burnin     = 200,
                        nsamples   = 800)

Without side information Macau is equivalent to standard Bayesian Matrix Factorization (BPMF).
However, if available using side information can significantly improve the model accuracy.
In the case of IC50 data the accuracy improves from RMSE of 0.90 to close to 0.60.
