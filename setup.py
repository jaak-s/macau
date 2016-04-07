import sys
import numpy as np
from glob import glob

from distutils.core import setup
from distutils.command.build_clib import build_clib
from distutils.extension import Extension
from distutils.errors import DistutilsSetupError
from Cython.Distutils import build_ext
from Cython.Build import cythonize
from distutils.sysconfig import get_python_inc

from distutils import log

import os
from textwrap import dedent

# for downloading Eigen
import tempfile
import urllib
import tarfile
import shutil

# checking out libfastsparse
import subprocess

## how to test -fopenmp: https://github.com/hickford/primesieve-python/blob/master/setup.py
def is_openblas_installed():
    """check if the C module can be build by trying to compile a small 
    program against the libyaml development library"""

    import tempfile
    import shutil

    import distutils.sysconfig
    import distutils.ccompiler
    from distutils.errors import CompileError, LinkError

    libraries = ['openblas', 'gfortran']

    # write a temporary .cpp file to compile
    c_code = dedent("""
    #include <cblas.h>

    int main(int argc, char* argv[])
    {
        int n = 3;
        int nn = n*n;
        double *A = new double[nn];
        double *C = new double[nn];
        A[0] = 0.5; A[1] = 1.4; A[2] = -0.1;
        A[3] = 2.3; A[4] = -.4; A[5] = 19.1;
        A[6] = -.72; A[7] = 0.6; A[8] = 12.3;

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,n,n,n,1,A, n, A, n, 0 ,C, n);

        return 0;
    }
    """)
    c_code_lapack = dedent("""
    #include <stdio.h>
    extern "C" void dpotrf_(char *uplo, int *n, double *a, int *lda, int *info);
    extern "C" void dpotrs_(char *uplo, int* n, int* nrhs, double* A, int* lda, double* B, int* ldb, int* info);

    int main(int argc, char* argv[])
    {
        int n = 3;
        int nn = n*n;
        int nrhs = 2;
        int info;
        char lower = 'L';
        double *A = new double[nn];
        double *B = new double[nrhs*nn];
        A[0] = 6.1;   A[1] = -0.65; A[2] = 5.1;
        A[3] = -0.65; A[4] = 2.4;   A[5] = -0.4;
        A[6] = 5.1;   A[7] = -0.4;  A[8] = 12.3;
        B[0] = 5.2; B[1] = -0.4;
        B[2] = 1.0; B[3] = 1.3;
        B[4] = 0.2; B[5] = -0.15;

        dpotrf_(&lower, &n, A, &n, &info);
        if(info != 0){ printf("c++ error: Cholesky decomp failed"); }
        dpotrs_(&lower, &n, &nrhs, A, &n, B, &n, &info);
        if(info != 0){ printf("c++ error: Cholesky solve failed"); }

        return 0;
    }
    """)
    tmp_dir = tempfile.mkdtemp(prefix = 'tmp_blas_')
    bin_file_name = os.path.join(tmp_dir, 'test_blas')
    file_name = bin_file_name + '.cpp'
    with open(file_name, 'w') as fp:
        fp.write(c_code)

    lapack_bin_file_name = os.path.join(tmp_dir, 'test_lapack')
    lapack_file_name = lapack_bin_file_name + '.cpp'
    with open(lapack_file_name, 'w') as fp:
        fp.write(c_code_lapack)

    # and try to compile it
    compiler = distutils.ccompiler.new_compiler(verbose=5)
    assert isinstance(compiler, distutils.ccompiler.CCompiler)
    distutils.sysconfig.customize_compiler(compiler)
    ldirs = ["/opt/OpenBLAS/lib", "/usr/local/lib", "/usr/lib/openblas-base"]

    try:
        compiler.link_executable(
            compiler.compile([file_name]),
            bin_file_name,
            libraries = libraries,
            library_dirs = ldirs,
            target_lang = "c++"
        )
    except CompileError:
        print('libopenblas compile error (please install OpenBLAS)')
        ret_val = False
    except LinkError:
        print('libopenblas link error (please install OpenBLAS)')
        ret_val = False
    else:
        ret_val = True

    try:
        compiler.link_executable(
            compiler.compile([lapack_file_name]),
            lapack_bin_file_name,
            libraries = libraries,
            library_dirs = ldirs,
            target_lang = "c++"
        )
    except CompileError:
        print('libopenblas lapack compile error (please install OpenBLAS)')
        ret_val = False
    except LinkError:
        print('libopenblas lapack link error (please install OpenBLAS)')
        ret_val = False
    else:
        ret_val = True

    shutil.rmtree(tmp_dir)
    return ret_val

def download_eigen_if_needed():
    url = "http://bitbucket.org/eigen/eigen/get/3.3-beta1.tar.bz2"
    eigen_inner = "eigen-eigen-ce5a455b34c0"
    dest = "lib/eigen3"
    if os.path.isdir(dest + "/Eigen"):
        return
    print("Downloading Eigen (v3.3)...")
    tmpdir = tempfile.mkdtemp()
    bzfile = tmpdir + "/3.3-beta1.tar.bz2"
    urllib.urlretrieve(url, bzfile)
    print("Download complete. Extracting Eigen ...")
    tf = tarfile.open(bzfile, "r:bz2")
    if not os.path.exists(dest):
        os.makedirs(dest)
    tf.extractall(path = tmpdir)
    print("Extracting complete.")
    tmpeigen = tmpdir + "/" + eigen_inner
    shutil.move(tmpeigen + "/Eigen", dest)
    shutil.move(tmpeigen + "/unsupported", dest)
    ## deleting tmp
    shutil.rmtree(tmpdir)

def checkout_libfastsparse():
    if os.path.exists("lib/libfastsparse/csr.h") or os.path.exists("lib/libfastsparse/LICENSE"):
        return
    print("Checking out git submodules (libfastsparse).")
    status = subprocess.call(["git", "submodule", "update", "--init", "--recursive"])
    if status != 0:
        raise RuntimeError("Could not checkout submodule. Please run 'git submodule update --init --recursive'.")
    print("Checking out completed.")

class build_clibx(build_clib):
    def build_libraries(self, libraries):
        for (lib_name, build_info) in libraries:
            sources = build_info.get('sources')
            sources = list(sources)
            if sources is None or not isinstance(sources, (list, tuple)):
                raise DistutilsSetupError(
                       "in 'libraries' option (library '%s'), "
                       "'sources' must be present and must be "
                       "a list of source filenames" % lib_name)

            log.info("building '%s' library", lib_name)
            include_dirs = build_info.get('include_dirs')
            extra_compile_args = build_info.get('extra_compile_args')
            extra_link_args = build_info.get('extra_link_args')
            objects = self.compiler.compile(sources,
                                            output_dir=self.build_temp,
                                            include_dirs=include_dirs,
                                            extra_preargs=extra_compile_args,
                                            extra_postargs=extra_link_args,
                                            debug=self.debug,
                                            )

            #self.compiler.link_shared_object(objects,
            #                                 'lib' + lib_name + '.so',
            #                                 output_dir=self.build_clib,
            #                                 extra_preargs=extra_compile_args,
            #                                 extra_postargs=extra_link_args,
            #                                 debug=self.debug)

            #lib_build_paths.append(('broot/lib', [self.build_clib + '/lib' + lib_name + '.so']))
            self.compiler.create_static_lib(objects, lib_name,
                                            output_dir = self.build_clib,
                                            debug=self.debug)

inc = ['lib/macau-cpp', 'lib/eigen3', 'lib/libfastsparse', np.get_include(), get_python_inc(), "/usr/local/include"]

libmacau = ('macau-cpp', dict(
    package='macau',
    sources = filter(lambda a: a.find("tests.cpp") < 0, glob('lib/macau-cpp/*.cpp')),
    include_dirs = inc,
    extra_compile_args = ['-fopenmp', '-O3', '-fstrict-aliasing', '-std=c++11'],
    #extra_link_args = ['-fopenmp'],
    language = "c++"
    ))

ext_modules=[
    Extension("macau", 
              sources = ["python/macau/macau.pyx", "python/macau/myblas.cpp"],
              include_dirs = inc,
              libraries = ["openblas", "pthread"],
              extra_compile_args = ['-std=c++11', '-fopenmp'],
              extra_link_args = ['-fopenmp'],
              language = "c++")
]

def main():
    if not is_openblas_installed():
        print("OpenBLAS not found. Please install.")
        sys.exit(1)
    else:
        print("OpenBLAS found.")
    download_eigen_if_needed()
    checkout_libfastsparse()

    setup(
        name = 'macau',
        version = "0.2",
        requires = ['numpy', 'scipy', 'cython(>=0.21)', 'cysignals', 'pandas'],
        libraries = [libmacau],
        packages = ["macau"],
        package_dir = {'' : 'python'},
        author = "Jaak Simm",
        url = "http://google.com",
        license = "MIT",
        author_email = "jaak.simm@gmail.com",
        cmdclass = {'build_clib': build_clibx, 'build_ext': build_ext},
        ext_modules = cythonize(ext_modules, include_path=sys.path)
    )

if __name__ == '__main__':
    main()

