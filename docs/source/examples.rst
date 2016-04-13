.. role:: python(code)
   :language: python

Examples
===========
In this examples we use ChEMBL IC50 dataset for proteins. The IC50 values and ECFP fingerprints can be downloaded from these two urls:
.. code-block:: bash

    wget http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm
    wget http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm

Matrix Factorization with Side Information
-------------------------------------------

Next we factorize the `compound x protein` IC50 matrix by using side information on the compounds
.. code-block:: python

    import macau
    import scipy.io

    ## loading data
    ic50 = scipy.io.mmread("/home/jaak/Dropbox/two/matrix-factorization/chembl_19_mf1c/chembl-IC50-346targets.mm")
    ecfp = scipy.io.mmread("/home/jaak/Dropbox/two/matrix-factorization/chembl_19_mf1c/chembl-IC50-compound-feat.mm")

    ## running factorization (Macau)
    result = macau.macau(Y = ic50, Ytest = 0.2, side=[ecfp, None], num_latent = 32, precision=5.0, burnin = 50, nsamples = 200)

In this example we have used 20% of the data for test by setting :python:`Ytest = 0.2` and use burn-in of 50 samples and then collect 200 samples of the model.
For obtaining higher accuracy, try `burnin = 200, nsamples = 800`.

The parameter :python:`side=[ecfp, None]` gets vector of two objects, the side information for rows and columns, respectively.
In this example we only use side information for the compounds.

The :python:`precision=5.0` specifies the precision of the IC50 observations, i.e., 1 / variance.


