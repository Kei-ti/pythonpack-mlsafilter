from __future__ import with_statement, print_function, absolute_import

from setuptools import setup, find_packages, Extension
from distutils.version import LooseVersion

import sys
import os
from glob import glob
from os.path import join

from setuptools.command.build_ext import build_ext as _build_ext

#DOCLINES = __doc__.split('\n')
_VERSION = '0.1'

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

#mlsa_src_top = "lib"
#mlsa_sources = glob(join(mlsa_src_top, "*.cpp"))

ext_modules = [
    Extension(
        name="pymlsa",
        #include_dirs=[mlsa_src_top],
        sources=[join("pymlsa", "pymlsa.pyx")], #+ mlsa_sources,
        language="c++")]

setup(
    name="pymlsa",
    #description=DOCLINES,
    #long_description='\n'.join(DOCLINES[2:]),
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    version=_VERSION,
    packages=find_packages(),
    setup_requires=[
        'numpy',
    ],
    install_requires=[
        'numpy',
        'cython>=0.24.0',
    ],
    extras_require={
        'test': ['nose'],
        'sdist': ['numpy', 'cython'],
    },
    author="Pymlsafilter Contributors",
    author_email="kou.tanaka.4research@gmail.com",
    url="https://github.com/Kei-ti/pythonpack-mlsafilter",
    keywords=['vocoder'],
    classifiers=[],
)
