#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/*")
    os.system("rm -rf dist/transito*")
    sys.exit()

# Load the __version__ variable without importing the package already
exec(open('transito/version.py').read())

# DEPENDENCIES
# 1. What are the required dependencies?
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()
# 2. What dependencies required to run the unit tests? (i.e. `pytest --remote-data`)
tests_require = ['pytest', 'pytest-cov', 'pytest-remotedata', 'codecov',
                 'pytest-doctestplus', 'codacy-coverage']
# 3. What dependencies are required for optional features?
# `BoxLeastSquaresPeriodogram` requires astropy>=3.1.
# `interact()` requires bokeh>=1.0, ipython.
# `PLDCorrector` requires pybind11, celerite.
extras_require = {"all": ["astropy>=3.1",
                           "bokeh>=1.0", "ipython",
                           "pybind11", "celerite"],
                  "test": tests_require}

setup(name='transito',
      version=__version__,
      description="A package for time series photometry "
                  "in Python.",
      long_description=open('README.rst').read(),
      author='Jaime Andrés Alvarado Montes',
      author_email='jaime-andres.alvarado-montes@hdr.mq.edu.au',
      url='https://docs.lightkurve.org',
      license='MIT',
      package_dir={
            'transito': 'transito'},
      packages=['transito'],
      install_requires=install_requires,
      extras_require=extras_require,
      setup_requires=['pytest-runner'],
      tests_require=tests_require,
      include_package_data=True,
      classifiers=[
          "Development Status :: 1 - Production/Stable",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research",
          "Topic :: Scientific/Engineering :: Astronomy",
          ],
      )
