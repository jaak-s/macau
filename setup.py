import sys
import numpy as np
from glob import glob

from distutils.core import setup
from distutils.command.build_clib import build_clib
from distutils.extension import Extension
from distutils.errors import DistutilsSetupError
from Cython.Distutils import build_ext
from distutils.sysconfig import get_python_inc

from distutils import log

## how to test -fopenmp: https://github.com/hickford/primesieve-python/blob/master/setup.py

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

inc = ['lib/macau-cpp', 'lib/eigen3', np.get_include(), get_python_inc(), "/usr/local/include"]

libmacau = ('macau-cpp', dict(
    package='macau',
    sources = glob('lib/macau-cpp/*.cpp'),
    include_dirs = inc,
    extra_compile_args = ['-fopenmp', '-O3', '-fstrict-aliasing', '-std=c++11'],
    extra_link_args = ['-fopenmp'],
    language = "c++"
    ))

ext_modules=[
    Extension("macau", 
              sources = ["python/macau/macau.pyx"],
              include_dirs = inc,
              extra_compile_args = ['-std=c++11', '-fopenmp'],
              extra_link_args = ['-fopenmp'],
              language = "c++")
]

def main():
    setup(
        name = 'macau',
        version = "0.2",
        requires = ['numpy', 'scipy', 'cython'],
        libraries = [libmacau],
        packages = ["macau"],
        package_dir = {'' : 'python'},
        author = "Jaak Simm",
        url = "http://google.com",
        license = "MIT",
        author_email = "jaak.simm@gmail.com",
        cmdclass = {'build_clib': build_clibx, 'build_ext': build_ext},
        ext_modules = ext_modules
    )

if __name__ == '__main__':
    main()

