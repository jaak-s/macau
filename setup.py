import sys
from glob import glob

from distutils.core import setup
from distutils.command.build_clib import build_clib
from distutils.extension import Extension
from Cython.Distutils import build_ext

## how to test -fopenmp: https://github.com/hickford/primesieve-python/blob/master/setup.py

libmacau = ('macau-cpp', dict(
    package='macau',
    sources = glob('lib/macau-cpp/hello.cpp'),
    include_dirs = ['lib/macau-cpp', 'lib/eigen3'],
    extra_compile_args = ['-fopenmp', '-O3', '-fstrict-aliasing', '-std=c++11'],
    extra_link_args = ['-fopenmp'],
    language = "c++"
    ))

ext_modules=[
    Extension("macau", 
              sources = ["python/macau/macau.pyx"],
              include_dirs = ["lib/macau-cpp"],
              extra_compile_args = ['-std=c++11'],
              language = "c++")
]

def main():
    setup(
        name = 'macau',
        version = "0.2",
        libraries = [libmacau],
        packages = ["macau"],
        package_dir = {'' : 'python'},
        author = "Jaak Simm",
        url = "http://google.com",
        license = "MIT",
        author_email = "jaak.simm@gmail.com",
        cmdclass = {'build_clib': build_clib, 'build_ext': build_ext},
        ext_modules = ext_modules
    )

if __name__ == '__main__':
    main()

