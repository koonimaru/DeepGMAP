===========================
README for DeepGMAP (0.0.0)
===========================
Time-stamp: <2018-02-15 13:55:42 Koh>

Introduction
============
One of the fundamental problems in biology is the genotype-phenotype mapping, or to predict a phenotype from a genotype. Genotype originally means \
a set of protein-coding genes that affect phenotypes, but here I extend the definition of genotype to genome sequences that determine phenotypes, \
which includes non-coding regions such as gene regulatory sequences. Currently, predicting gene regulatory regions from genome sequences is a \
challenging task. DeepGMAP is a Deep learning-based Genotype MAPping platform to predict them. It can train different architectures of neural \
networks with epigenomic data, and "map" potential gene regulatory regions on a genome.

A related paper has been published in XX.

Install
=======

Please check the file 'INSTALL' in the distribution.

Usage of deepgmap
=================

::

  deepgmap [-h] [--version]
             {pridict,train,generate_input,genome_divide}

:Example for enhancer prediction: "deepgmap predict -i ./data/outputs/XXXX.meta -o ./data/predictions/ -b ./data/inputs/mm10_dnase_subset/XXXX.bed.labeled -t ./data/test_genome/mm10_window1000_stride200_chr2_*.npz"

:Example for training a model: "deepgmap train -i ./data/inputs/mm10_dnase_subset/ -c conv4frss -o ./data/outputs/ -G 1"

There are five functions currently available.

:train:				Train a model with your data set. The model can be chosen by the option '-c'. deepsea, basset, danq, frss4, frss3 are available choices, but you can also create a new model.
:predict:			Predict regulatory sequences in a genome or test a newly trained model.
:generate_input:	Generate a training data set that is randomly shuffled and distributed into mini-batches.
:generate_test:		Generate a test data set, or convert a genome sequence that you want to annotate its regulatory regions into input data set 
:genome_divide:		This function creates input files for "generate_input" and "generate_test" function. The genomic data of humans and mice is already included in this package under XX directory. If you have a de novo genome sequence or want to try other species to train a model.  



1. To annotate regulatory regions in a genome with a trained model. 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a de novo genome sequence and want to annotate their regulatory sequences with an already trained model, you need to convert AGCT sequences to onehot vectors.
Firstly, the multiple fasta file of your genome should be divided into a particular window size and stride by the following command:

deepgmap genome_divide -i ./data/genomes/mm10.fa -w 1000 -s 500

, which produces mm10_window1000_stride500.bed and mm10_window1000_stride500.fa (you need to change mm10.fa to your multiple fasta file).
The next step is to convert AGCT symbols to matrices of onehot arrays by the following command:

deepgmap generate_test -i ./data/genomes/mm10_window1000_stride500.fa -o ./data/test_genome/mm10_window500_stride500_ -t 16 -C all

, which produces a series of npz files. To predict regulatory sequences, type

deepgmap predict -i ./data/outputs/conv4frssplus2_Mon_May__7_092815_2018.ckpt-127179.meta -o ./data/predictions/ -t ./data/test_genome/mm10_window1000_stride200*.npz -G 0


Output files
~~~~~~~~~~~~

1. narrowPeak files are tabular files that contain genomic regions with prediction score from 0 to 1. 
   The files can be visualized with the IGV genome viewer (http://software.broadinstitute.org/software/igv/).
   Each file is corresponding to one of labels you have trained your model with.
2. A npz file is a numpy array of prediction scores. The array coordinate is the same as the labeled file.


2. To train a model with epigenomic data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to train a model with your data set, first, you need to generate genomic sequences labeled with your data. To do so, run the following command: 

deepgmap generate_input -b ./data/inputs/mm10_ctcf/*.narrowPeak -g ./data/genomes/mm10_window1000_stride500 -p ctcf_all -t 16 -s 100 -r 0.30 -d chip-seq

This command would take 10 min to a few hours depending on your machine and the amount of data. If you see a memory error, you can reduce the RAM usage by increasing the 
integer of -n option. It is not the optimal algorithm, will be improved in the future. Next, to train a model run 

deepgmap train -i ./data/genomes/mm10_window1000_stride500s100r30_train_data_set/ -c conv4frss -o ./data/outputs/

, where -i option is to feed a training data set, -c to specify a model type, -o to specify the output directory, and -G to specify index of GPUs (optional). For model types, 
currently deepsea, basset, danq, dandqblock, conv4, conv3frss, conv4frss, conv4frsspluss, are available.
   

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
Run:
predict -i ./data/outputs/conv4frss_<date>.ckpt-<train step>.meta -o ./data/predictions/ -b ./data/genomes/mm10_window1000_stride500.bed.labeled 
-t ./data/test_data/mm10_window1000_stride500_chr2_*.npz

Output files
~~~~~~~~~~~~
1. narrowPeak files are tabular files that contain genomic regions with prediction score from 0 to 1. 
   The files can be visualized with the IGV genome viewer (http://software.broadinstitute.org/software/igv/).
   Each file is corresponding to one of labels you have trained your model with.
2. A npz file is a numpy array of prediction scores. The array coordinate is the same as the labeled file.
3. A log file that contains AUROC and AUPRC scores.
4. A pdf file of ROC and PRC.

