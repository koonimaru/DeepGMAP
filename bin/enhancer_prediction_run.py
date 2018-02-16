#!/usr/bin/python
# Time-stamp: <2018-0-0 0:0:0 Koh Onimaru>

"""Description: enhancer_prediction main executable.
Copyright (c) 20XX Koh Onimaru <koh.onimaru@gmail.com>
This code is free software; you can redistribute it and/or modify it
under the terms of the BSD License (see the file COPYING included with
the distribution).
@status: release candidate
@version: $Id$
@author:  Koh Onimaru
@contact: koh.onimaru@gmail.com
"""

# ------------------------------------
# python modules
# ------------------------------------

import os
import sys
import argparse as ap
import tempfile

VERSION=""

def prepare_argparser ():
    """Prepare optparser object. New options will be added in this
    function first.   
    """
    
    description = "%(prog)s -- regulatory sequence prediction"
    epilog = "For command line options of each command, type: %(prog)s COMMAND -h"
    argparser = ap.ArgumentParser( description = description, epilog = epilog ) #, usage = usage )
    argparser.add_argument("--version", action="version", version="%(prog)s "+VERSION)
    subparsers = argparser.add_subparsers( dest = 'subcommand_name' ) #help="sub-command help")
    
    # command for 'train'
    add_train_parser( subparsers )
    
    # command for 'predict'
    add_predict_parser( subparsers )
    
    # command for 'generate_input'
    add_generate_input_parser( subparsers )
    
   
    return argparser

def add_train_parser( subparsers ):
    """Add main function 'train' argument parsers.
    """
    
    """
    try:
        options, args =getopt.getopt(sys.argv[1:], 'm:i:n:b:o:c:p:', ['mode=', 'input_dir=', 'loop_num=', 'test_batch_num=', 'output_dir=','network_constructor=','pretrained_model='])
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(2)
    if len(options)<3:
        print('too few argument')
        sys.exit(0)
    for opt, arg in options:
        if opt in ('-m', '--mode'):
            mode=arg
        elif opt in ('-i', '--input_dir'):
            input_dir=arg
            if input_dir.endswith("/"):
                input_dir=str(input_dir)+"*"
            elif input_dir.endswith("*"):
                pass
            else:
                input_dir=str(input_dir)+"/*"
                
        elif opt in ('-n', '--loop_num'):
            loop_num_=int(arg)
        elif opt in ('-b', '--test_batch_num'):
            test_batch_num=int(arg)
        elif opt in ('-o', '--output_dir'):
            output_dir=arg
        elif opt in ('-c', '--network_constructor'):
            model_name=arg
        elif opt in ('-p', '--pretrained_model'):
            pretrained_dir=arg
    """
    
    
    argparser_train = subparsers.add_parser("train", help="Train a network with genomic sequences.")
    
    # group for input files
    group_input = argparser_train.add_argument_group( "Input arguments" )
    group_input.add_argument( "-i", "--in_dir", dest = "in_directory", type = str, required = True,
                              help = "The directory of a traing data set. Labeled and shuffled genomic seuqences in a format of numpy array, produced by 'input_generate' command. REQUIRED." )
    group_input.add_argument( "-c", "--network_constructor", dest = "model",required = True, type = str,
                                    help = "The name of a model to train. Model files should be in network_constructors directory. REQUIRED.")
    group_input.add_argument( "-m", "--mode", dest = "mode", type = str,  default = "train",
                              choices = ("train", "pretrain"),
                              help = "Training mode. \"train\", \"pretrain\". If users want to retrain a model, select pretrain mode. But, pretrain is still in prep, Default: train")
    group_input.add_argument( "-n", "--loop_num", dest = "loop_number", type = int, default = None,
                              help = "The number of mini-batches to train. If not provided, the program will go through all mini-batches (i.e. all npz files) except test batches." )
    group_input.add_argument( "--test_batch_num", dest = "test_batch_number", type = int, default = None,
                              help = "A file number for test batches. If not provided, the program automatically select the last three batches in a series of npz files." )
    group_input.add_argument( "-p", "--pretrained_model", dest = "ckpt_file", type = str, default = None,
                              help = "the ckpt file of pretrained model. If \"pretrain\" mode is selected, this option is recquired." )
    group_input.add_argument( "-G", "--GPU", dest = "GPU_number", type = int,default = 1,
                              help = "The number of GPU on your machine. Default: 1" )
    group_input.add_argument( "-k", "--max_to_keep", dest = "max_to_keep", type = int,default = 2,
                              help = "The number of trained variables to keep. During training, the program saves a set of variables when the test accuracy is good enough. Default: 2" )
    # group for output files
    group_output = argparser_train.add_argument_group( "Output arguments" )
    #add_outdir_option( group_output )
    group_output.add_argument( "-o", "--out_dir", dest = "out_directory", type = str, required = True,
                               help = "An output directory. REQUIRED.")

    
    return


def add_predict_parser( subparsers ):
    """
    
    options, args =getopt.getopt(sys.argv[1:], 'i:o:n:b:t:g:c:G:T', ['input_dir=','output_dir=','network_constructor=','bed=', 'test_genome=','genome_bed=','chromosome=','GPU=', 'TEST='])
    TEST=True
    path_sep=os.path.sep
    chromosome_of_interest='chr2'
    output_dir='./'
    model_name=""
    bed_file=None
    max_to_keep=2
    GPU=1
    for opt, arg in options:
        if opt in ('-i', '--input_dir'):
            input_dir=arg
            if not os.path.isfile(input_dir):
                print(input_dir+' does not exist')
                sys.exit(0)
        elif opt in ('-o', '--output_dir'):
            output_dir=arg
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except OSError as exc:
                    if exc.errno != exc.errno.EEXIST:
                        raise

        elif opt in ('-n', '--network_constructor'):
            model_name=arg
            #if not os.path.isfile(model_name):
                #print(model_name+' does not exist')
                #sys.exit(0)
        elif opt in ('-b', '--bed'):
            bed_file=arg
            if not os.path.isfile(bed_file):
                print(bed_file+' does not exist')
                sys.exit(0)
        elif opt in ('-t', '--test_genome'):
            test_genome=arg
            #if not os.path.isfile(test_genome):
                #print(test_genome+' does not exist')
                #sys.exit(0)
        elif opt in ('-g','--genome_bed'):
            genome_bed=arg
            if not os.path.isfile(genome_bed):
                print(genome_bed+' does not exist')
                sys.exit(0)
        elif opt in ('-C','--chromosome'):
            chromosome_of_interest=arg
            if chromosome_of_interest=="all":
                TEST=False
                
        elif opt in ('-G','--GPU'):
            GPU=int(arg)
    """
    
    
    argparser_predict = subparsers.add_parser( "predict",
                                                 help = "Predict regulatory sequences" )
    argparser_predict.add_argument( "-i", "--in_file", dest = "input_ckpt", type = str, required = True,
                                      help = "A ckpt-xxxxx.meta file (an output file from training, which contains information of trained variables). REQUIRED." )
    argparser_predict.add_argument( "-o", "--out_dir", dest = "out_directory", type = str, required = True,
                                      help = "an output directory for prediction results. REQUIRED.")
    argparser_predict.add_argument( "-c", "--network_constructor", dest = "model", type = str,
                                      help = "The name of a model to train. Model files should be in network_constructors directory. If not specified, the program automatically infer it from the ckpt_file." )
    argparser_predict.add_argument( "-b", "--bed", dest = "labeled_bed_file", type = str, required = True,
                                      help = "A labeled_bed_file (.bed.labeled), which can be created by 'generate_input' command, or included in the data directory." )
    argparser_predict.add_argument( "-t", "--test_genome", dest = "test_genome_files", type = str, required = True,
                                      help = "Test genome files, e.g. /path/to/test_files/*.npz. Files can be created by 'generate_test' command. REQUIRED." )
    argparser_predict.add_argument("-G," "--GPU", dest = "GPU_number", type = int, default = 1,
                                      help = "The number of GPU in your machine. Currently, the program can use only one GPU. So, multiple GPU won't speed up. Default: 1" )
    argparser_predict.add_argument( "-C", "--chromosome", dest = "chromosome", type = str, default = "chr2",
                                      help = "Set a target chromosome or a contig for prediction. Default: chr2" )
    argparser_predict.add_argument( "-T", "--TEST", dest = "test_or_prediction", type = str, default = True,
                                      help = "True or False. If True, the program will create ROC plots by comparing with labeled_bed_file. Default: True" )
    return

def add_generate_input_parser( subparsers ):
    """"

    options, args =getopt.getopt(sys.argv[1:], 'b:g:w:t:p:s:r:h:', ['bed=', 'genome=', 'window_size=' 'threads=', 'prefix=','sample_number=','reduce_genome=','help='])


    window_size=None
    threads=1
    pref=''
    sample_num=100
    reduce_genome=0.75
    howto=\
    "usage: input_generator_from_narrowPeaks [-h] [-b FILE_DIR] [-g FILE_DIR] \n\
    [-t INT] [-s INT] [-r FLOAT] [-p PREFIX] \n\
    \n\
    optional arguments:\n\
      -h, --help            show this help message and exit\n\
      --version             show program's version number and exit\n\
    Input files arguments:\n\
      -b FILE_DIR, --bed FILE_DIR\n\
                  A narrowPeak file directory. REQUIRED.\n\
      -g FILE_DIR, --genome FILE_DIR\n\
                  The directory plus prefix of a bed file and a fasta file that are \n\
                  binned with a particular window size. If you have \n\
                  /path/to/mm10_1000.bed and /path/to/mm10_1000.fa,the input would \n\
                  be '-g /path/to/mm10_1000'. REQUIRED.\n\
      -t INT, --threads INT\n\
                  The number of threads. Multithreading is performed only when \n\
                  saving output numpy arrays. Default: 1\n\
      -s INT, --sample_number INT\n\
                  The number of samples in a mini-batch. Default: 100\n\
      -r FLOAT, --reduse_genome FLOAT\n\
                  A fraction to ignore signal-negative genome sequences. Default: \n\
                  0.75\n\
    Output arguments:\n\
      -p PREFIX, --prefix PREFIX\n\
                  The prefix of output files and folders. Default: ''\n"
    """
    argparser_generate_input = subparsers.add_parser( "generate_input",
                                                   help = "Generate a train data set from narrowPeak files." )
    argparser_generate_input.add_argument( "-b", "--bed", dest = "in_directory", type = str, required = True,
                                        help = "A directory that contains narrowPeak or bed files. \
                                        These files, which indicate signal peak regions in a genome, can be created \
                                        by MACS2 peak caller or other softwares, or can be downloaded from ENCODE project. REQUIRED" )
    argparser_generate_input.add_argument( "-g", "--genome" , dest = "genome_file_prefix", type = str, required = True,
                                        help = "The directory plus prefix of a bed file and a fasta file that are\
                                        binned with a particular window size. If you have path/to/mm10_1000.bed and\
                                         /path/to/mm10_1000.fa,the input would be '-g /path/to/mm10_1000'. REQUIRED." )
    argparser_generate_input.add_argument( "-t", "--threads", dest = "thread_number", type = int,
                                       help = "The number of threads. Multithreading is performed only when saving output numpy arrays. Default: 1", default = 1 )
    argparser_generate_input.add_argument( "-s", "--sample_number", dest = "sample_number", type = int,
                                       help = "The number of samples in a mini-batch. Default: 100", default = 100 )
    argparser_generate_input.add_argument( "-r", "--reduse_genome", dest="genome_fraction", help="A fraction to ignore signal-negative genome sequences. Default: 0.75",default=0.75)
    argparser_generate_input.add_argument("-p", "--prefix", dest="out_prefix",
                                        help = "The prefix of output files and folders. Default: ''", default = '' )
    argparser_generate_input.add_argument("-C", "--chromosome", dest="chromosome_to_skip",
                                        help = "A chromosome or contig name to skip for training. The name must be one of the chromosome/contig names in the input genome file. Default: chr2", default = 'chr2' )
    return

def add_generate_test_parser( subparsers ):

    """
        try:
        options, args =getopt.getopt(sys.argv[1:], 'i:t:o:p:', ['input_dir=','target_chr=', 'output_dir=','process='])
    except getopt.GetoptError as err:
        print str(err)
        sys.exit(2)
    if len(options)<3:
        print('too few argument')
        sys.exit(0)
        
    threads=psutil.cpu_count()
    
    for opt, arg in options:
        if opt in ('-i', '--input_dir'):
            input_file=arg
        elif opt in ('-t', '--target_chr'):
            target_chr=arg
        elif opt in ('-o', '--output_dir'):
            output_file=arg
        elif opt in ('-p', '--process'):
            threads=int(arg)
    """


    argparser_generate_test = subparsers.add_parser( "generate_test",
                                                    help = "Generate a data set for a test or an application of a trained model." )
    argparser_generate_test.add_argument( "-i", "--in_file", dest = "input_genome" , type = str, required = True,
                                         help = "A multiple fasta file containing genome DNA sequences. REQUIRED" )
    argparser_generate_test.add_argument("-C", "--chromosome", dest = "chromosome", type = str, default = "chr2",
                                      help = "Set a target chromosome or a contig for prediction. Default: chr2" )
    argparser_generate_test.add_argument( "-o", "--out_dir", dest = "out_directory", type = str, required = True,
                                         help = "")
    argparser_generate_test.add_argument( "-t", "--threads", dest = "thread_number", type = int,
                                       help = "The number of threads. Multithreading is performed only when saving output numpy arrays. Default: 1", default = 1 )
  
    return
    

# ------------------------------------
# Main function
# ------------------------------------
def main():
    """The Main function/pipeline for enhancer_prediction.
    
    """
    # Parse options...
    argparser = prepare_argparser()
    args = argparser.parse_args()


    subcommand  = args.subcommand_name

    if subcommand == "train":
        # train a model
        from enhancer_prediction.train.deepshark_local_oop_1d import run
        run( args )
    elif subcommand == "predict":
        # predict regulatory sequences or test a trained model
        from enhancer_prediction.post_train_tools.trained_deepshark_local2 import run
        run( args )
    elif subcommand == "generate_input":
        # generate a train data set
        from enhancer_prediction.data_preprocessing_tools.input_generator_from_narrowPeaks import run
        run( args )
    elif subcommand == "generate_test":
        from enhancer_prediction.post_train_tools.inputfileGeneratorForGenomeScan_p import run
        run( args )
if __name__ == '__main__':
    print os.getcwd()
    print(sys.path)
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupted me! ;-) Bye!\n")
        sys.exit(0)