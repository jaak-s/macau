# Macau-cpp - Bayesian Factorization with Side Information
Highly optimized and parallelized implementation of Bayesian Factorization called Macau.

# Examples
For examples see [documentation](http://macau.readthedocs.io/en/latest/source/examples.html).

# Installation on Ubuntu
```bash
# install dependencies:
sudo apt-get install python-pip python-numpy python-scipy python-pandas cython
sudo apt-get install libopenblas-dev autoconf gfortran
sudo pip install cysignals

# checkout and install Macau
git clone https://github.com/jaak-s/macau.git
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
git clone https://github.com/jaak-s/macau.git
cd macau
CC=g++-5 CXX=g++-5 python setup.py install
```

# Contributors
In alphabetical order:
- Tom Vander Aa
- Tom Haber
- Jaak Simm 

