# Macau-cpp - Bayesian Factorization with Side Information
Highly optimized and parallelized methods for Bayesian Factorization, including BPMF and Macau.
The package uses optimized OpenMP/C++ code with a Cython wrapper to factorize large scale matrices.
Macau method is able to perform **matrix** and **tensor** factorization while incorporating high-dimensional side information to the factorization.

# Examples
For examples see [documentation](http://macau.readthedocs.io/en/latest/source/examples.html).

# Installation
To install Macau it possible to use pre-compiled binaries or compile it from source.

## Source installation on Ubuntu
```bash
# install dependencies:
sudo apt-get install python-pip python-numpy python-scipy python-pandas cython
sudo apt-get install libopenblas-dev autoconf gfortran

# checkout and install Macau
git clone https://github.com/jaak-s/macau.git
cd macau
python setup.py install --user
```

## Installing with pip
If you have openblas installed (package libopenblas-dev in Ubuntu) available and `gcc` and `g++` installed,
then following steps install macau:
```bash
git clone https://github.com/jaak-s/macau.git
cd macau
pip install .
```

Instead of `pip install .` one can use
```bash
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
brew install homebrew/core/openblas
brew install gcc

# checkout and install Macau
git clone https://github.com/jaak-s/macau.git
cd macau
CC=g++-5 CXX=g++-5 python setup.py install
```

## Docker
Macau is also available using Docker image at `stadius/macau`.

Without mounting a local directory the docker can be executed by
```bash
docker run -it --rm -p 8888:8888 stadius/macau
```

To mount a local directory add `-v ~/my_data_dir:/data` where
`~/my_data_dir` is on the local system and `/data` will be the folder
in the container:
```bash
docker run -v ~/my_data_dir:/data -it --rm -p 8888:8888 stadius/macau
```

## Binary installion on Ubuntu
There is a plan to support Python wheel packages. Currently, we do not have one built yet.

# Contributors
- Jaak Simm (Macau C++ version, Cython wrapper, Macau MPI version, Tensor factorization)
- Adam Arany (Probit noise model)
- Tom Vander Aa (OpenMP optimized BPMF)
- Tom Haber (Original BPMF code)
