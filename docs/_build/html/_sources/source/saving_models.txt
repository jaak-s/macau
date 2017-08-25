.. role:: python(code)
   :language: python

Saving models
==================
To save samples of the Macau model you can add :python:`save_prefix = "mymodel"` when calling :python:`macau`.
This option will store all samples of the latent vectors, their mean vectors and link matrices to the disk.
Additionally, the global mean-value that Macau adds to all predictions is also stored.

Example
-------------------------------------------

.. code-block:: python
   :emphasize-lines: 16

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
                        burnin     = 100,
                        nsamples   = 500,
                        save_prefix = "chembl19")

Saved files
-------------------------------------------
The saved files for sample :python:`N` for the rows are

- Latent vectors :python:`chembl19-sampleN-U1-latents.csv`.
- Latent means: :python:`chembl19-sampleN-U1-latentmeans.csv`.
- Link matrix (beta): :python:`chembl19-sampleN-U1-link.csv`.
- Global mean value: :python:`chembl19-meanvalue.csv` (same for all samples).

Equivalent files for the column latents are stored in :python:`U2` files.

Using the saved model to make predictions
-----------------------------------------
These files can be loaded with numpy and used to make predictions.

.. code-block:: python

   import numpy as np

   ## global mean value (common for all samples)
   meanvalue = np.loadtxt("chembl19-meanvalue.csv").tolist()

   ## loading sample 1
   N = 1
   U = np.loadtxt("chembl19-sample%d-U1-latents.csv" % N, delimiter=",")
   V = np.loadtxt("chembl19-sample%d-U2-latents.csv" % N, delimiter=",")

   ## predicting Y[0, 7] from sample 1
   Yhat_07 = U[:,0].dot(V[:,7]) + meanvalue

   ## predict the whole matrix from sample 1
   Yhat = U.transpose().dot(V) + meanvalue

Note that in Macau the final prediction is the average of the predictions from all samples.
This can be accomplished by looping over all of the samples and averaging the predictions.

Using the saved model to predict new rows (compounds)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here we show an example how to make a new prediction for a compound (row) that was not in the dataset, by using its side information and saved link matrices.

.. code-block:: python
   :emphasize-lines: 11

   import numpy as np
   import scipy.io

   ## loading side info for arbitrary compound (can be outside of the training set)
   xnew = scipy.io.mmread("chembl-IC50-compound-feat.mm").tocsr()[17,:]

   ## loading sample 1
   meanvalue = np.loadtxt("chembl19-meanvalue.csv").tolist()
   N = 1
   lmean = np.loadtxt("chembl19-sample%d-U1-latentmean.csv" % N, delimiter=",")
   link  = np.loadtxt("chembl19-sample%d-U1-link.csv" % N,       delimiter=",")
   V     = np.loadtxt("chembl19-sample%d-U2-latents.csv" % N,    delimiter=",")

   ## predicted latent vector for xnew from sample 1
   uhat = xnew.dot(link.transpose()) + lmean

   ## use predicted latent vector to predict activities across columns 
   Yhat = uhat.dot(V) + meanvalue

Again, to make good predictions you would have to change the example to loop over all of the samples (and compute the mean of Yhat's).

Tensor models
~~~~~~~~~~~~~
As in the matrix case the tensor factorization can be saved using :python:`save_prefix` argument
and later loaded from disk to make predictions.
To make predictions we recall that the value of a tensor model is given by a tensor contraction of all latent matrices. Specifically, the prediction for the element :python:`Yhat[i,j,k]` of a rank-3 tensor is given by

.. math::

   \hat{Y}_{ijk} = \sum_{d=1}^D u^{(1)}_{d,i} u^{(2)}_{d,j} u^{(3)}_{d,k} + mean

Next we show how to compute this prediction using :python:`numpy`.
Assuming we have run and saved a model named :python:`save_prefix = "mytensor"` of tensor of rank :python:`3`
we can load the latent matrices and make predictions using :python:`np.einsum` function.

.. code-block:: python

   import numpy as np

   ## global mean value (common for all samples)
   meanvalue = np.loadtxt("mytensor-meanvalue.csv").tolist()

   ## loading latent matrices for sample 1
   N = 1
   U1 = np.loadtxt("mytensor-sample%d-U1-latents.csv" % N, delimiter=",")
   U2 = np.loadtxt("mytensor-sample%d-U2-latents.csv" % N, delimiter=",")
   U3 = np.loadtxt("mytensor-sample%d-U3-latents.csv" % N, delimiter=",")

   ## predicting Y[7, 0, 1] from sample 1
   Yhat_701 = sum(U1[:,7] * U2[:,0] * U3[:,1]) + meanvalue

   ## predict the whole tensor from sample 1, using np.einsum
   Yhat = np.einsum(U1, [0, 1], U2, [0, 2], U3, [0, 3]) + meanvalue


As before this is a prediction from a single sample. For better predictions we should loop over all of the samples
and average their predictions (their Yhat's).

It is also possible to predict only **slices** of the full tensors using :python:`np.einsum`:

.. code-block:: python

   ## predict the slice Y[7, :, :] from sample 1
   Yhat_7xx = np.einsum(U1[:,7], [0], U2, [0, 2], U3, [0, 3]) + meanvalue

   ## predict the slice Y[:, 0, :] from sample 1
   Yhat_x0x = np.einsum(U1, [0, 1], U2[:,0], [0], U3, [0, 3]) + meanvalue

   ## predict the slice Y[:, :, 1] from sample 1
   Yhat_xx1 = np.einsum(U1, [0, 1], U2, [0, 2], U3[:,1], [0]) + meanvalue

All 3 examples above give a matrix (rank-2 tensor) as a result.
To get the prediction for a slice we replaced the full latent matrix (:python:`U1`) with a single specific latent vector (:python:`U1[:,7]`) and changed its indexing from :python:`[0, 1]` to :python:`[0]` as the indexing now over a vector.
