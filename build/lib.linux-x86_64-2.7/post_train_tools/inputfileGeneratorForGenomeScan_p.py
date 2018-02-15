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

import psutil
import getopt


def DNA_to_array_converter(input_file,target_chr):
    seq_list=[]
    position_list=[]
    
    
    with open(input_file, 'r') as fin:
    
        SEQ=False
        if target_chr=="all":
            for line in fin:
                if line.startswith('>'):
                    if not "_" in line and not line.startswith('>chrM'):
                        print line,
                        position_list.append(line.strip('\n'))
                        SEQ=True
                    else:
                        SEQ=False
                elif SEQ:
                    line=line.strip('\n')
                    data_width=len(line)
                    sequence=np.reshape(sb2.AGCTtoArray3(line,data_width), (1,data_width,4,1))
                    seq_list.append(sequence)
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
                    sequence=np.reshape(sb2.AGCTtoArray3(line,data_width), (1,data_width,4,1))
                    seq_list.append(sequence)

    return position_list, seq_list
        

def array_saver(outfile,positions,sequences):
    print('saving '+outfile)
    np.savez_compressed(outfile,positions=positions,sequences=sequences)
        
def run(args):
    
    main()

def main():
    
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
                                      position_list[i*chunk_num+l*threads:(i+1)*chunk_num+l*threads], 
                                      seq_list[i*chunk_num+l*threads:(i+1)*chunk_num+l*threads])))
        for j in jobs:
            j.start()
            
        for j in jobs:
            j.join()
        

    
if __name__== '__main__':
    main()
    
    
    
    
    
    
    
    
    
    