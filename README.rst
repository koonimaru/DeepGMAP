===================
README for DeepGMAP
===================

Introduction
============
One of the fundamental goals in biology is the genotype-phenotype mapping, or to predict a phenotype from a genotype. Genotype originally means \
a set of protein-coding genes that affect phenotypes, but here I extend the definition of genotype to genome sequences that determine phenotypes, \
which includes non-coding regions such as gene regulatory sequences. Currently, predicting gene regulatory regions from genome sequences is a \
challenging task. DeepGMAP is a Deep learning-based Genotype MAPping platform to predict them. It can train different architectures of neural \
networks with epigenomic data, and "map" potential gene regulatory regions on a genome.

A related paper has been posted at bioRxiv, https://doi.org/10.1101/355974.

Install
=======

Please check the file 'INSTALL' in the distribution.

Usage of deepgmap
=================

::

  deepgmap [-h] [--version]
             {pridict,train,generate_input,genome_divide}

:Example for enhancer prediction: deepgmap predict -i /path/to/conv4frssXXXX.meta -o /path/to/predictions/ -t /path/to/test_data/mm10_window1000_stride300

:Example for training a model: deepgmap train -i /path/to/inputs/mm10_dnase_subset/ -c conv4frss -o /path/to/outputs/

There are five functions currently available.

:train:				Train a model with your data set. The model can be chosen by the option '-c'. deepsea, basset, danq, conv4, conv4frss, conv3frss are available choices, but you can also create a new model.
:predict:			Predict regulatory sequences in a genome or test a newly trained model.
:generate_input:	Generate a training data set that is randomly shuffled and distributed into mini-batches.
:generate_test:		Generate a test data set, or convert a genome sequence that you want to annotate its regulatory regions into input data set 
:genome_divide:		This function creates input files for "generate_input" and "generate_test" function. The genomic data of humans and mice is already included in this package under XX directory. If you have a de novo genome sequence or want to try other species to train a model.  



1. To annotate regulatory regions in a genome with a trained model. 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a de novo genome sequence and want to annotate their regulatory sequences with an already trained model, you need to convert AGCT sequences to onehot vectors.
Firstly, the multiple fasta file of your genome should be divided into a particular window size and stride by the following command::

 $ deepgmap genome_divide -i /path/to/genomes/mm10.fa -w 1000 -s 300

This produces mm10_window1000_stride300.bed and mm10_window1000_stride300.fa (you need to change mm10.fa to your multiple fasta file).
The next step is to convert AGCT symbols to matrices of onehot arrays::

 $ deepgmap generate_test -i /path/to/genomes/mm10_window1000_stride300 -o /path/to/test_data/mm10_window1000_stride300 -t 16 -C all


To predict regulatory sequences, type::

 $ deepgmap predict -i /path/to/outputs/conv4frss_Tue_May_21_004410_2019/train.ckpt-26259/train.ckpt-26259.meta -o /path/to/predictions/ -t /path/to/data/test_data/mm10_window1000_stride300


Output files
~~~~~~~~~~~~

1. narrowPeak files are tabular files that contain genomic regions with prediction score from 0 to 1. 
   The files can be visualized with the IGV genome viewer (http://software.broadinstitute.org/software/igv/).
   Each file is corresponding to one of labels you have trained your model with.
2. A npz file is a numpy array of prediction scores. The array coordinate is the same as the labeled file.


2. To train a model with epigenomic data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to train a model with your data set, first, you need to generate genomic sequences labeled with your data::

 $ deepgmap generate_input -b /path/to/data/inputs/mm10_ctcf -g /path/to/data/genomes/mm10_window1000_stride300 -p ctcf_test -t 8

This command would take 10 min to a few hours depending on your machine and the amount of data. If you see a memory error, you can reduce the RAM usage by increasing the 
integer of -n option. It is not the optimal algorithm, will be improved in the future. Next, to train a model run::

 $ deepgmap train -i /path/to/data/inputs/mm10_ctcf/ctcf_test_mm10_window1000_stride300s100r0.8_train_data_set -c conv4frss -o /path/to/data/outputs/

The "-i" option is to feed a training data set, -c to specify a model type, -o to specify the output directory, and -G to specify index of GPUs (optional). Currently available models are  
deepsea, basset, danq, danqblock, conv4, conv3frss, conv4frss, conv4frsspluss.
   

Output files
~~~~~~~~~~~~

1. A set of tensorflow output files: "conv4frss_<date>.ckpt-<train step>.meta", "conv4frss_<date>.ckpt-<train step>.index", 
"conv4frss_<date>.ckpt-<train step>.data-00000-of-00001", "checkpoint", and additional similar files that were saved during training when train 
accuracy was high. These files contains trained variables, and are required for running "deepgmap predict".  
2. A log file named conv4frss_<date>.log, which contains information about the run command, model name, several hyper parameters, and input files and so on.
3. A plot file in pdf format, named "conv4frss_<date>_plot.pdf". It shows the progress of training with F1 values in the top panel and cost in the bottom.
4. A set of trained variables in numpy array format, named "conv4frss_<date>_trained_variables.npz". It contains almost same information with the tensorflow outputs.
It is just for convenience to analyze trained models.


3. To test a trained model with test data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run::

 $ deepgmap predict -i /path/to/data/outputs/conv4frss_<date>.ckpt-<train step>.meta -o /path/to/data/predictions/ -b ./data/inputs/mm10_ctcf/ctcf_test_mm10_window1000_stride300.bed.labeled \
 -t /path/to/data/test_data/mm10_window1000_stride300_chr2

Alternatively, you can test a trained model just by giving the directory of a train output with option -l::

 $ deepgmap predict -l /path/to/data/outputs/conv4frss_<date>
 
In this case, the prediction result is generated under the input directory. 

Output files
~~~~~~~~~~~~
1. narrowPeak files are tabular files that contain genomic regions with prediction score between 0 and 1. 
   The files can be visualized with the IGV genome viewer (http://software.broadinstitute.org/software/igv/).
   Each file is corresponding to one of labels you have trained your model with.
2. A npz file is a numpy array of prediction scores. The array coordinate is the same as the labeled file.
3. A log file that contains AUROC and AUPRC scores.
4. A pdf file of ROC and PRC.

Examples of running a docker image
======================

 $ docker run -v $HOME:$HOME --runtime=nvidia -it --rm koonimaru/deepgmap deepgmap genome_divide -i /path/to/genomes/mm10.fa -w 1000 -s 300

 $ docker run -v $HOME:$HOME --runtime=nvidia -it --rm koonimaru/deepgmap deepgmap train -i /full/path/to/mm10_ctcf/ctcf_mm10_window1000_stride300s100r0.8_train_data_set -o /full/path/to/outputs -c conv4frss

 $ docker run -v $HOME:$HOME --runtime=nvidia -it --rm koonimaru/deepgmap deepgmap predict -l /full/path/to/output_directory_of_train -t /full/path/to/mm10_window1000_stride300

If you are running docker through qsub, remove "-it".
