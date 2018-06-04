==========================
INSTALL Guide For DeepGMAP
==========================

Prerequisites
=============

DeepGMAP is verified to work only on Linux (Ubunru 16.10). 

Python version 2.7.

Numpy_ (>=1.6). 

Cython_ (>=0.18) is an optional requirement to recompile ``.pyx`` files.

Tensorflow_ (>=1.8)

Scikitlearn_ (>=0.19.1)

matplotlib_

bedtools_ (>=2.25)

.. _Numpy: http://www.scipy.org/Download
.. _Cython: http://cython.org/
.. _Tensorflow: https://www.tensorflow.org/	
.. _Scikitlearn: http://scikit-learn.org/
.. _matplotlib: https://matplotlib.org/
.. _bedtools: http://bedtools.readthedocs.io/

Download source
===============

 $ git clone https://github.com/koonimaru/DeepGMAP.git

Configure enviroment variables
==============================

You need to add the downloaded location (in this example home directory; $HOME) to your ``PYTHONPATH`` and ``PATH`` environment variables.

PYTHONPATH
~~~~~~~~~~

You need to include the new value in your ``PYTHONPATH`` by
adding this line to your ``~/.bashrc``::

 $ export PYTHONPATH=$HOME/DeepGMAP/:$PYTHONPATH

Then, type::

 $ source .bashrc

Or, re-login to your account.

PATH
~~~~

You'll also like to add a new value to your
PATH environment variable so that you can use the deepgmap command line
directly::

 $ export PATH=$HOME/DeepGMAP/bin/:$PATH

--
Koh Onimaru <koh.oinmaru@gmail.com>

