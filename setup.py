from __future__ import with_statement, print_function, absolute_import

from setuptools import setup, find_packages, Extension
from distutils.version import LooseVersion

import sys
import os
from glob import glob
from os.path import join

from setuptools.command.build_ext import build_ext as _build_ext

import builtins  # ★追加

_VERSION = '0.1'

class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        builtins.__NUMPY_SETUP__ = False  # ★修正（__builtins__ ではなく builtins）
        import numpy
        self.include_dirs.append(numpy.get_include())

ext_modules = [
    Extension(
        name="pymlsa",
        sources=[join("pymlsa", "pymlsa.pyx")],
        language="c++"
    )
]

setup(
    name="pymlsa",
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