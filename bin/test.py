#!/usr/bin/python2
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
    #Check community site: http://groups.google.com/group/macs-announcement/
    #Source code: https://github.com/taoliu/MACS/"
    # top-level parser
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
    group_input.add_argument( "-i", "--input_dir", dest = "in_directory", type = str, required = True,
                              help = "The directory of a traing data set. Labeled and shuffled genomic seuqences in a format of numpy array, produced by 'input_generate' command. REQUIRED." )
    group_input.add_argument( "-c", "--network_constructor", dest = "py_file",required = True, type = str,
                                    help = "The name of a model to train. Model files should be in network_constructors directory. REQUIRED.")
    group_input.add_argument( "-m", "--mode", dest = "mode", type = str,
                              choices = ("train", "pretrain"),
                              help = "Training mode. \", \"train\", \"pretrain\". If users want to retrain a model, select pretrain mode. But, pretrain is still in prep, Default: train",
                              default = "train" )
    group_input.add_argument( "-n", "--loop_num", dest = "n_int", type = int, default = None,
                              help = "The number of mini-batches to train. If not provided, the program will go through all mini-batches (i.e. all npz files) except test batches." )
    group_input.add_argument( "-b","--test_batch_num", dest = "b_int", type = int, default = None,
                              help = "A file number for test batches. If not provided, the program automatically select the last three batches in a series of npz files." )
    group_input.add_argument( "-p", "--pretrained_model", dest = "ckpt_file", type = str,
                              help = "the ckpt file of pretrained model. If \"pretrain\" mode is selected, this option is recquired." )
    group_input.add_argument( "-G", "--GPU", dest = "g_int", type = int,default = 1,
                              help = "The number of GPU on your machine. Default: 1" )
    # group for output files
    group_output = argparser_train.add_argument_group( "Output arguments" )
    #add_outdir_option( group_output )
    group_output.add_argument( "-o", "--output_dir", dest = "out_directory", type = str, required = True,
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
    argparser_predict.add_argument( "-i", "--input_file", dest = "ckpt_file", type = str, required = True,
                                      help = "A ckpt-xxxxx.meta file (an output file from training, which contains information of trained variables). REQUIRED." )
    argparser_predict.add_argument( "-o", "--output_dir", dest = "directory", type = str, required = True,
                                      help = "an output directory for prediction results. REQUIRED.")
    argparser_predict.add_argument( "-c", "--network_constructor", dest = "py_file", type = str,
                                      help = "The name of a model to train. Model files should be in network_constructors directory. If not specified, the program automatically infer it from the ckpt_file." )
    argparser_predict.add_argument( "-b", "--bed", dest = "labeled_bed_file", type = str, required = True,
                                      help = "A labeled_bed_file (.bed.labeled), which can be created by 'generate_input' command, or included in the data directory." )
    argparser_predict.add_argument( "-t", "--test_genome", dest = "test_genome_files", type = str, required = True,
                                      help = "Test genome files, e.g. /path/to/test_files/*.npz. Files can be created by 'generate_test' command. REQUIRED." )
    argparser_predict.add_argument("-G," "--GPU", dest = "INT", type = int, default = 1,
                                      help = "The number of GPU in your machine. Currently, the program can use only one GPU. So, multiple GPU won't speed up. Default: 1" )
    argparser_predict.add_argument( "-C", "--chromosome", dest = "chromosome", type = str, default = "chr2",
                                      help = "Set a target chromosome or a contig for prediction. Default: chr2" )
    argparser_predict.add_argument( "-T", "--TEST", dest = "boolian", type = str, default = "True",
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
    argparser_bdgpeakcall = subparsers.add_parser( "generate_input",
                                                   help = "Generate a train data set from narrowPeak files." )
    argparser_bdgpeakcall.add_argument( "-b", "--bed", dest = "directory", type = str, required = True,
                                        help = "A directory that contains narrowPeak or bed files. \
                                        These files, which indicate signal peak regions in a genome, can be created \
                                        by MACS2 peak caller or other softwares, or can be downloaded from ENCODE project. REQUIRED" )
    argparser_bdgpeakcall.add_argument( "-g", "--genome" , dest = "genome_file_prefix", type = str, required = True,
                                        help = "The directory plus prefix of a bed file and a fasta file that are\
                                        binned with a particular window size. If you have path/to/mm10_1000.bed and\
                                         /path/to/mm10_1000.fa,the input wouldbe '-g /path/to/mm10_1000'. REQUIRED." )
    argparser_bdgpeakcall.add_argument( "-t", "--threads", dest = "int", type = int,
                                       help = "The number of threads. Multithreading is performed only when saving output numpy arrays. Default: 1", default = 1 )
    argparser_bdgpeakcall.add_argument( "-s", "--sample_number", dest = "int", type = int,
                                       help = "The number of samples in a mini-batch. Default: 100", default = 100 )
    argparser_bdgpeakcall.add_argument( "-r", "--reduse_genome", dest="float", help="A fraction to ignore signal-negative genome sequences. Default: 0.75",default=0.75)
    argparser_bdgpeakcall.add_argument("-p", "--prefix", dest="PREFIX", action="store_true",
                                        help = "The prefix of output files and folders. Default: ''", default = '' )
    return

def add_generate_test_parser( subparsers ):
    """Add function 'broad peak calling on bedGraph' argument parsers.
    """
    argparser_bdgbroadcall = subparsers.add_parser( "bdgbroadcall",
                                                    help = "Call broad peaks from bedGraph output. Note: All regions on the same chromosome in the bedGraph file should be continuous so only bedGraph files from MACS2 are accpetable." )
    argparser_bdgbroadcall.add_argument( "-i", "--ifile", dest = "ifile" , type = str, required = True,
                                         help = "MACS score in bedGraph. REQUIRED" )
    argparser_bdgbroadcall.add_argument( "-c", "--cutoff-peak", dest = "cutoffpeak", type = float,
                                         help = "Cutoff for peaks depending on which method you used for score track. If the file contains qvalue scores from MACS2, score 2 means qvalue 0.01. DEFAULT: 2",
                                         default = 2 )
    argparser_bdgbroadcall.add_argument( "-C", "--cutoff-link", dest = "cutofflink", type = float,
                                         help = "Cutoff for linking regions/low abundance regions depending on which method you used for score track. If the file contains qvalue scores from MACS2, score 1 means qvalue 0.1, and score 0.3 means qvalue 0.5. DEFAULT: 1", default = 1 )
    argparser_bdgbroadcall.add_argument( "-l", "--min-length", dest = "minlen", type = int,
                                         help = "minimum length of peak, better to set it as d value. DEFAULT: 200", default = 200 )
    argparser_bdgbroadcall.add_argument( "-g", "--lvl1-max-gap", dest = "lvl1maxgap", type = int,
                                         help = "maximum gap between significant peaks, better to set it as tag size. DEFAULT: 30", default = 30 )
    argparser_bdgbroadcall.add_argument( "-G", "--lvl2-max-gap", dest = "lvl2maxgap", type = int,
                                         help = "maximum linking between significant peaks, better to set it as 4 times of d value. DEFAULT: 800", default = 800)

    #add_outdir_option( argparser_bdgbroadcall )    
    #add_output_group( argparser_bdgbroadcall )    
    return
    

# ------------------------------------
# Main function
# ------------------------------------
def main(args=None):
    
    print args
    """The Main function/pipeline for enhancer_prediction.
    
    """
    # Parse options...
    argparser = prepare_argparser()
    args = argparser.parse_args()
    print argparser.print_help()
    print args
    print args.out_directory
    print args.subcommand_name
    


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.stderr.write("User interrupted me! ;-) Bye!\n")
        sys.exit(0)