# Use an official Python runtime as a base image
FROM jupyter/scipy-notebook

MAINTAINER Jaak Simm <jaak.simm@gmail.com>

USER root

RUN apt-get update && \
    apt-get -yq install libopenblas-dev autoconf gfortran && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


USER $NB_USER

# Install Python 3 macau
RUN mkdir $HOME/git && \
    cd $HOME/git && \
    git clone https://github.com/jaak-s/macau.git && \
    cd $HOME/git/macau && \
    git checkout v0.5.0 && \
    python3 setup.py install --user && \
    cd $HOME && rm -rf git

ADD python/macau/examples/Macau-with-ChEMBL.ipynb $HOME/work/

