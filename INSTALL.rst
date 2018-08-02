==========================
INSTALL Guide For DeepGMAP
==========================

Prerequisites
=============

DeepGMAP is verified to work on Linux (Ubunru 16.10). Also using GPU is highly recommended. 

Python version 2.7.

Numpy_ (>=1.6). 

Cython_ (>=0.18) is an optional requirement to recompile ``.pyx`` files.

Tensorflow_ (>=1.8) Note that Tensorflow requires cuDNN and cudna libraries. 

Scikitlearn_ (>=0.19.1)

matplotlib_

bedtools_ (>=2.25)

.. _Numpy: http://www.scipy.org/Download
.. _Cython: http://cython.org/
.. _Tensorflow: https://www.tensorflow.org/	
.. _Scikitlearn: http://scikit-learn.org/
.. _matplotlib: https://matplotlib.org/
.. _bedtools: http://bedtools.readthedocs.io/

Installing tensorflow-gpu
=========================
To accelerate computing, I highly recommend you to use `cuda-enabled GPUs`_.tensorflow-gpu itself can be easily installed with pip 
(sudo pip install tensorflow-gpu or sudo pip install -r requirements.txt). But, to make tensorflow-gpu work, you need the right versions 
and right combination of cuDNN and cuda toolkit libraries (please check the `tensorflow web site`_). 

.. _cuda-enabled GPUs: https://developer.nvidia.com/cuda-gpus
.. _tensorflow web site: https://www.tensorflow.org/install/install_linux

Download source and data
========================
To download the source code from our github repository::

 $ git clone https://github.com/koonimaru/DeepGMAP.git
 
To download a trial data set::

 $ wget https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/12257744/DeepGMAPdatalight.tar.gz

Then, decompress it with "tar -xvzf DeepGMAPdatalight.tar.gz", and place the folder named "data" under the DeepGMAP directory.
 
Configure environment variables
===============================

You need to add the downloaded location (in this example home directory: $HOME) to your ``PYTHONPATH`` and ``PATH`` environment variables.

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

