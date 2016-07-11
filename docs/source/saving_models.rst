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

Similarily, the :python:`link` matrices can be loaded. Here we show an example how to make a new prediction for a compound that was not in the dataset.

.. code-block:: python

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
