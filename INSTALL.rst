==========================
INSTALL Guide For DeepGMAP
==========================

Prerequisites
=============

DeepGMAP is verified to work only in Linux (Ubunru 16.10). 

Python version 2.7.

Numpy_ (>=1.6). 

Cython_ (>=0.18) is an optional requirement to recompile ``.pyx`` files.

Tensorflow_ (>=1.8)

matplotlib_

bedtools_ (>=2.25)

.. _Numpy: http://www.scipy.org/Download
.. _Cython: http://cython.org/

Install from source
===================

To install a source distribution of DeepGMAP, go to the directory of
DeepGMAP, and run the install script::

 $ python setup.py install --prefix $HOME


Configure enviroment variables
==============================

After running the setup script, you might need to add the install
location to your ``PYTHONPATH`` and ``PATH`` environment variables. The
process for doing this varies on each platform, but the general
concept is the same across platforms.

PYTHONPATH
~~~~~~~~~~

You need to include the new value in your ``PYTHONPATH`` by
adding this line to your ``~/.bashrc``::

 $ export PYTHONPATH=$HOME/DeepGMAP/:$PYTHONPATH


PATH
~~~~

You'll also like to add a new value to your
PATH environment variable so that you can use the deepgmap command line
directly::

 $ export PATH=$HOME/DeepGMAP/bin/:$PATH

--
Koh Onimaru <koh.oinmaru@gmail.com>

