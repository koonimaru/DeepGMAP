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

On Linux, using bash, I include the new value in my ``PYTHONPATH`` by
adding this line to my ``~/.bashrc``::

 $ export PYTHONPATH=/home/taoliu/lib/python2.7/site-packages:$PYTHONPATH


PATH
~~~~

Just like your ``PYTHONPATH``, you'll also need to add a new value to your
PATH environment variable so that you can use the MACS command line
directly. Unlike the ``PYTHONPATH`` value, however, this time you'll need
to add ``PREFIX/bin`` to your PATH environment variable. The process for
updating this is the same as described above for the ``PYTHONPATH``
variable::

 $ export PATH=/home/taoliu/bin:$PATH

--
Koh Onimaru <koh.oinmaru@gmail.com>

