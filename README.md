# Macau-cpp - Bayesian Factorization with Side Information
Highly optimized and parallelized methods for Bayesian Factorization, including BPMF and Macau. The package uses optimized OpenMP/C++ code with a Cython wrapper to factorize large scale matrices. Macau method provides also the ability to incorporate high-dimensional side information to the factorization.

# Examples
For examples see [documentation](http://macau.readthedocs.io/en/latest/source/examples.html).

# Installation
To install Macau it possible to use pre-compiled binaries or compile it from source.

## Source installation on Ubuntu
```bash
# install dependencies:
sudo apt-get install python-pip python-numpy python-scipy python-pandas
sudo apt-get install libopenblas-dev autoconf gfortran
pip install cython --user --upgrade

# checkout and install Macau
git clone https://github.com/jaak-s/macau.git
cd macau
python setup.py install --user
```

## Source installation on Mac
```bash
# install dependencies
pip install numpy
pip install scipy
pip install pandas
pip install cython
# install brew (http://brew.sh/)
brew install homebrew/science/openblas
brew install gcc

# checkout and install Macau
git clone https://github.com/jaak-s/macau.git
cd macau
CC=g++-5 CXX=g++-5 python setup.py install
```

## Binary installion on Ubuntu
There is a plan to support Python wheel packages. Currently, we do not have one built yet.

# Contributors
- Jaak Simm (Macau C++ version, Cython wrapper, Macau MPI version)
- Tom Vander Aa (OpenMP optimized BPMF)
- Tom Haber (Original BPMF code)

