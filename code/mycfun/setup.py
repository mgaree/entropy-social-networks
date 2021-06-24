"""Build with `python setup.py build_ext --inplace`"""

import os
import io
import sys

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext


class build_ext_with_numpy(build_ext):
    def run(self):
        import numpy
        self.include_dirs.append(numpy.get_include())
        build_ext.run(self)


extensions = [Extension("cdataproc",
                        [os.path.join('cdataproc.c')])]

try:
    import numpy
except ImportError:
    setup_requires = ['numpy<1.16']
else:
    setup_requires = []

setup(name='cdataproc',
      description='C versions of data processing for entropy experiment',
      setup_requires=setup_requires ,
      install_requires=['numpy'],
      author='Michael Garee',
      ext_modules=extensions,
      cmdclass={'build_ext': build_ext_with_numpy})
