# Macau-cpp - Bayesian Factorization with Side Information
Macau is highly optimized and parallelized implementation of Bayesian Factorization.

Macau trains a Bayesian model for **collaborative filtering** by also incorporating side information on rows and/or columns to improve the accuracy of the predictions.
Macau employs Gibbs sampling to sample both the latent vectors and the link matrix that connects the side information to the latent vectors. Macau supports **high-dimensional** side information (e.g., millions of features) by using **conjugate gradient** based noise injection sampler.

# Installation on Ubuntu
```bash
# install dependencies:
sudo apt-get install python-pip python-numpy python-scipy python-pandas cython
sudo apt-get install libopenblas-dev autoconf gfortran
sudo pip install cysignals

# checkout and install Macau
git clone --recursive https://github.com/jaak-s/macau.git
cd macau
sudo python setup.py install
```

# Installation on Mac
```bash
# install dependencies
pip install numpy
pip install scipy
pip install pandas
pip install cython
pip install cysignals
# install brew (http://brew.sh/)
brew install homebrew/science/openblas
brew install gcc

# checkout and install Macau
git clone --recursive https://github.com/jaak-s/macau.git
cd macau
CC=g++-5 CXX=g++-5 python setup.py install
```

# Contributors
In alphabetical order:
- Tom Vander Aa
- Tom Haber
- Jaak Simm 

