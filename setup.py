#!/usr/bin/env python3
#
# Copyright (c) 2015, Tinghui Wang <tinghui.wang@wsu.edu>
# All rights reserved.

from setuptools import setup, find_packages
from Cython.Build import cythonize
import os

CLASSIFIERS = """\
Development Status :: 2 - Pre-Alpha
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Operating System :: POSIX
Operating System :: POSIX :: Linux
Programming Language :: Python
Programming Language :: Python :: 3.5
Topic :: Home Automation
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Scientific/Engineering :: Information Analysis
""".splitlines()

NAME = "pyActLearn"
MAINTAINER = "Tinghui Wang (Steve)"
MAINTAINER_EMAIL = "tinghui.wang@wsu.edu"
DESCRIPTION = ("Activity Learning package designed for rapid prototyping of " +
               "activity learning algorithms used with WSU CASAS smart home datasets.")
LONG_DESCRIPTION = DESCRIPTION
LICENSE = "BSD"
URL = "https://github.com/TinghuiWang/pyActLearn"
DOWNLOAD_URL = ""
AUTHOR = "Tinghui Wang (Steve)"
AUTHOR_EMAIL = "tinghui.wang@wsu.edu"
PLATFORMS = ["Linux"]

# Get Version from pyActLearn.version
exec_results = {}
exec(open(os.path.join(os.path.dirname(__file__), 'pyActLearn/version.py')).read(), exec_results)
version = exec_results['version']


def do_setup():
    setup(
        name=NAME,
        version=version,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        classifiers=CLASSIFIERS,
        platforms=PLATFORMS,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        license=LICENSE,
        keywords=' '.join(['activity recognition', 'smart home', 'smart environment']),
        packages=find_packages('.'),
        entry_points={'console_scripts': ['casas_download = pyActLearn.bin.casas_download:main']},
        install_requires=['numpy>=1.7.1', 'scipy>=0.11', 'six>=1.9.0', 'theano>=0.8.0'],
        ext_modules=cythonize("pyActLearn/learning/*.pyx", gdb_debug=True)
    )

if __name__ == "__main__":
    do_setup()
