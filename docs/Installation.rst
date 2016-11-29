Installation
============

Install using pip
-----------------

You can install pyActLearn package using Python package manager ``pip``.
The package source codes are avaialble on github.

.. code-block:: bash

   $ pip3 install git+git://github.com/TinghuiWang/pyActLearn.git \
     -r https://raw.githubusercontent.com/TinghuiWang/pyActLearn/master/requirements.txt

.. note::

   At the moment, pyActLearn package only supports Python 3.
   In most Linux distributions, the Python 3 package manager is named as ``pip3``.

You can also add ``--user`` switch to the command to install the package in your home folder,
and ``--upgrade`` switch to pull the latest version from github.

.. warning::

   ``pip`` may try to install or update packages such as Numpy, Scipy and Theano if they are
   not present or outdated. If you want to use your system package manager such as ``apt`` or
   ``yum``, you can add ``--no-deps`` switch and install all the requirements manually.

Install in Develop Mode
-----------------------

To install the package in developer mode, first fork your own copy of pyActLearn on github.
Then, you can clone and edit your repository and install with ``setup.py`` script using
the following command (replace ``user`` with your own github user name).

.. code-block:: bash

   $ git clone https://github.com/USER/pyActLearn.git
   $ python3 setup.py develop
