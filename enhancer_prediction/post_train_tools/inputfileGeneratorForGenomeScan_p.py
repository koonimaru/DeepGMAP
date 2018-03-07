import re
import numpy as np
import gzip
#output_handle=open("/home/koh/MLData/test.txt", 'w')
import cPickle
import time 
import gzip
import pickle
import math
import os.path
import multiprocessing
import sys
import glob
import enhancer_prediction.data_preprocessing_tools.seq_to_binary2 as sb2
import time
import psutil
import getopt


def DNA_to_array_converter(input_file,target_chr):
    seq_list=[]
    position_list=[]
    b1=0.0
    i=0
    with open(input_file, 'r') as fin:
    
        SEQ=False
        if target_chr=="all":
            for line in fin:
                if line.startswith('>'):
                    if not "_" in line and not line.startswith('>chrM'):
                        #print line,
                        position_list.append(line.strip('\n'))
                        SEQ=True
                    else:
                        SEQ=False
                    if i%100000==0:
                        print line
                elif SEQ:
                    line=line.strip('\n')
                    data_width=len(line)
                    a1=time.time()
                    seq_list.append(sb2.AGCTtoArray3(line,data_width))
                    b1+=time.time()-a1
                i+=1
                if i%100000==0:
                    print b1
                    sys.exit()
        else:
            for line in fin:
                if line.startswith('>'):
                    if line.startswith('>'+str(target_chr)+':'):
                        print line
                        position_list.append(line.strip('\n'))
                        SEQ=True
                    else:
                        SEQ=False
                elif SEQ:
                    line=line.strip('\n')
                    data_width=len(line)
                    #sequence=np.zeros([1,1000,4,1], np.int16)
                                       
                    seq_list.append(sb2.AGCTtoArray3(line,data_width))
                    
    return position_list, seq_list
        

def array_saver(outfile,positions,sequences):
    print('saving '+outfile)
    np.savez_compressed(outfile,positions=positions,sequences=sequences)
        
def run(args):
    
    main(args)

def main(args=None):
    if args is not None:
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
        """
        input_file=args.input_genome
        target_chr=args.chromosome
        output_file=args.out_directory
        threads=args.thread_number
        print args
    else:
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
    
        print options
    
    position_list, seq_list=DNA_to_array_converter(input_file,target_chr)
    seq_num=len(position_list)
    print seq_num
    
    DIVIDES_NUM=seq_num/120000

    if DIVIDES_NUM%threads==0:
        outerloop=DIVIDES_NUM/threads
    else:
        outerloop=DIVIDES_NUM/threads+1
        
        
    
    
    if seq_num%DIVIDES_NUM==0:
        chunk_num=seq_num/DIVIDES_NUM
    else:
        chunk_num=seq_num/DIVIDES_NUM+1
    if DIVIDES_NUM>=threads:
        job_num=threads
    else:
        job_num=DIVIDES_NUM
        
    print DIVIDES_NUM, threads, outerloop, job_num
    
    
    for l in range(outerloop):
        jobs = []    
        for i in range(job_num):
            if i*chunk_num+l*threads>seq_num:
                break
            jobs.append(multiprocessing.Process(target=array_saver, 
                                args=(str(output_file)+str(i+l*threads), 
                                      position_list[i*chunk_num+l*threads*chunk_num:(i+1)*chunk_num+l*threads*chunk_num], 
                                      seq_list[i*chunk_num+l*threads*chunk_num:(i+1)*chunk_num+l*threads*chunk_num])))
        for j in jobs:
            j.start()
            
        for j in jobs:
            j.join()
        

    
if __name__== '__main__':
    main()
    
    
    
    
    
    
    
    
    
    