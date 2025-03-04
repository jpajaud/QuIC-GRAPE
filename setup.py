import sys
# this line is a quick fix for an error in which the numpy package is not visible to the virtual environment
# sys.path.append('path/to/environment/site-packages')
from setuptools   import setup, find_packages
import numpy

setup(name = 'QuICGRAPE',
      version='0.0.0',
      description='Utilities for performing optimal control optimization',
      author='Jon Pajaud',
      author_email='jpajaud2@gmail.com',
      packages = find_packages(),
      package_data={'grape':['calc/*']},
      zip_safe=False,
      include_dirs=[numpy.get_include()],
)
