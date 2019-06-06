import sys
import getopt
import glob as glb
#from genome_labeling2 import genome_label2 as genome_label
import importlib as il

_genome_label=il.import_module("deepgmap.data_preprocessing_tools.genome_labeling2")
genome_label=_genome_label.genome_label2
import os
import subprocess as sp
#import deepgmap.data_preprocessing_tools.seq_to_binary2 as sb2
sb2=il.import_module("deepgmap.data_preprocessing_tools.seq_to_binary2")

#from inputfileGenerator_multiple_label3 import seqtobinarydict2 as seqtobinary
inputfileGenerator_multiple_label3=il.import_module("deepgmap.data_preprocessing_tools.inputfileGenerator_multiple_label3")
seqtobinary=inputfileGenerator_multiple_label3.seqtobinarydict2
#from inputfileGenerator_multiple_label3 import dicttoarray
#from inputfileGenerator_multiple_label3 import array_saver
array_saver=inputfileGenerator_multiple_label3.array_saver

import multiprocessing
import time 
import datetime
import numpy as np
import random
import math
#from deepgmap.data_preprocessing_tools.inputfileGenerator_multiple_label3 import array_saver_one_by_one
PATH_SEP=os.path.sep
def bedtools(cmd, tmpout):
    try:
        tmp_out=open(tmpout, 'w')
        sp.check_call(cmd, stdout=tmp_out)
        tmp_out.close()
    except:
        sys.exit("\nSomething went wrong while trying to run bedtools. Please check if bedtools is installed correctly.")
def getDupes_a(l):
    '''moooeeeep'''
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    for x in l:
        if x in seen or seen_add(x):
            yield x

def run(args):
    main(args)

def main(args=None):
    start=time.time()
    
    threads=1
    pref=''
    sample_num=100
    reduce_genome=0.75
    chr_to_skip="chr2"
    if args is not None:
        bed_file_dir=args.in_directory
        genome_pref=args.genome_file_prefix
        threads=args.thread_number
        sample_num=args.sample_number
        reduce_genome=args.genome_fraction
        pref=args.out_prefix
        chr_to_skip=args.chromosome_to_skip
        data_type=args.data_type
        chunck_data=args.chunck_data
    else:
            
        try:
            options, args =getopt.getopt(sys.argv[1:], 'b:g:w:t:p:s:r:h:', ['bed=', 'genome=', 'window_size=' 'threads=', 'prefix=','sample_number=','reduce_genome=','help='])
        except getopt.GetoptError as err:
            print(str(err))
            sys.exit(1)
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
        
        if len(options)<2:
            print('too few argument')
            print(howto)
            sys.exit(0)
        for opt, arg in options:
            if opt in ('-h', '--help'):
                print(howto)
            elif opt in ('-b', '--bed'):
                bed_file_dir=arg
            
            elif opt in ('-g', '--genome'):
                genome_pref=arg
                
    
            elif opt in ('-t', '--threads'):
                threads=int(arg)
            elif opt in ('-p', '--prefix='):
                pref=arg
            elif opt in ('-s', '--sample_number'):
                sample_num=int(arg)
            elif opt in ('-r', '--reduce_genome'):
                reduce_genome=float(arg)
                
    genome_1000=genome_pref+PATH_SEP+"genome.bed"
    genome_fasta=genome_pref+PATH_SEP+"genome.fa"
    if os.path.isfile(genome_1000)==False:
        print(genome_1000+" is missing.")
        sys.exit(1)
    if os.path.isfile(genome_fasta)==False:
        print(genome_fasta+" is missing.")
        sys.exit(1)
    
    with open(genome_1000, 'r') as fin:
        line=fin.readline()
        line=line.split()
        window_size=int(line[2])-int(line[1])
    bed_file_list=[]
    if bed_file_dir.endswith('.narrowPeak') or bed_file_dir.endswith('.bed'):
        bed_file_list=sorted(glb.glob(bed_file_dir))       
        out_dir=os.path.split(bed_file_dir)[0]+"/"
        
    else:
        if not bed_file_dir.endswith(PATH_SEP):
            bed_file_dir+=PATH_SEP
        bed_file_dir_=bed_file_dir+"*.narrowPeak"
        out_dir=bed_file_dir
        bed_file_list=sorted(glb.glob(bed_file_dir_))
        if len(bed_file_list)==0:
            bed_file_dir_=bed_file_dir+"*.bed"
            bed_file_list=sorted(glb.glob(bed_file_dir_))
            
    if len(bed_file_list)==0:
        sys.exit('no bed nor narrowPeak files in '+bed_file_dir)
        
    head, tail = os.path.split(genome_1000)
    labeled_genome=str(out_dir)+str(pref)+'_'+str(tail)+'.labeled'
    output_dir=str(out_dir)+str(pref)+'_'+str(tail.split('.')[0])+"s"+str(sample_num)+"r"+str(reduce_genome)+'_train_data_set'+PATH_SEP

    #create labeled genome file (.bed.labeled), only if it does not exist
    if not os.path.isfile(labeled_genome):
        print("reading "+str(len(bed_file_list))+" narrowPeak files...")
        if len(bed_file_list)==0:
            sys.exit("No peak files in "+str(bed_file_dir))
        bed_file_list_2=[]
        
        #preparing for parallel execution of bedtools
        jobs=[]
        #print(bed_file_list[0])
        h, t=os.path.split(bed_file_list[0])
        #print(bed_file_list[0])
        bed_dir=h+"/"+str(pref)+"_"+str(tail)+"_list"
        if not os.path.isdir(bed_dir):
            os.makedirs(bed_dir)
        for b in bed_file_list:
            h, t=os.path.split(b)
            b_=bed_dir+PATH_SEP+os.path.splitext(t)[0]+"_"+str(window_size)+".bed"
            if data_type=="dnase-seq":
                cmd=["bedtools", "intersect", "-u", "-a", str(genome_1000), "-b", str(b)]
            else:
                cmd=["bedtools", "intersect","-F", "0.4", "-f", "0.6", "-e", "-u", "-a", str(genome_1000), "-b", str(b)]
            #b_=os.path.splitext(b)[0]+"_"+str(window_size)+".bed"
            jobs.append(multiprocessing.Process(target=bedtools,args=(cmd,b_)))
            
            bed_file_list_2.append(b_)
        dups=list(getDupes_a(bed_file_list_2))
        if len(dups) is not 0:
            sys.exit(str(dups)+" are duplicated")
        job_num=len(jobs)
        job_loop=int(math.ceil(job_num/float(threads)))
        
        for jloop in range(job_loop):
            for thread in range(threads):
                if thread+threads*jloop<job_num:
                    jobs[thread+threads*jloop].start()
            for thread in range(threads):
                if thread+threads*jloop<job_num:
                    jobs[thread+threads*jloop].join()
        genome_label(bed_file_list_2, genome_1000,labeled_genome)
        #sys.exit()
    else:
        print('As '+labeled_genome +' already exists, skipping generating this file. \nIf you want to generate a new one, you need change the prefix or remove the old one.')

    
    if os.path.isfile(labeled_genome):
        with open(labeled_genome, 'r') as f1:
            f2=f1.readlines()
        
        
        label_genome_length=len(f2)
        print(label_genome_length)
        shuf=list(range(label_genome_length))
        random.shuffle(shuf)
        read_len=int(math.ceil(label_genome_length/float(chunck_data)))
        
        if "," in chr_to_skip:
            chr_to_skip=chr_to_skip.split(',')
        else:
            chr_to_skip=[chr_to_skip]
        
        print(read_len)
        for ooloop in range(chunck_data):
            sub_shuf=sorted(shuf[ooloop*read_len:(ooloop+1)*read_len])
            print(sub_shuf[0:10])
            f2_=[f2[f2s] for f2s in sub_shuf]
            label_position, label_list, skipped, pos_no=sb2.label_reader2(f2_, chr_to_skip, reduce_genome)
                #label_list=np.array(label_list, np.int8)
            with open(genome_fasta, 'r') as f1:
                binaryDNAdict, _ =seqtobinary(f1,label_position)
            #print(len(label_position), len(position))
            #neg_skipped=skipped
            dna_dict_length=len(binaryDNAdict)
            lnum=len(label_position)
            #if lnum==0:
                #print lnum, dna_dict_length
            
            pos_rate=np.round(pos_no/float(lnum), 3)*100
            neg_rate=100-pos_rate
            to_print1="\n"+str(skipped)+" negative sequences are skipped.\nThe rate of positives vs negatives is " + str(pos_rate)+":"+str(neg_rate)
            print(to_print1)
            del label_position
            #print("\t".join(label_position[:2])+"\n"+ "\t".join(label_position[:-2])+"\n"+"\t".join(position[:2])+"\n"+"\t".join(position[:-2]))
                    
               
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except OSError as exc:
                    if exc.errno != exc.errno.EEXIST:
                        raise
            
            #binaryDNAdict, label_list=dicttoarray(binaryDNAdict,position, label_list,label_position,reduce_genome)
            
            #binaryDNAdict=np.array(binaryDNAdict, np.int8)[shuf]
            #label_list=np.array(label_list, np.int8)[shuf]
            #binaryDNAdict_shuf=[]
            #binaryDNAdict_shuf_append=binaryDNAdict_shuf.append
            #label_list_shuf=[]
            #label_list_shuf_append=label_list_shuf.append
            shuf2=list(range(dna_dict_length))
            random.shuffle(shuf2)
           
            print("\nsaving train data set to "+output_dir+" with "+str(threads)+" threads")
            
            
            print(dna_dict_length, len(label_list))
            if dna_dict_length%threads==0:
                batch=dna_dict_length//threads
            else:
                batch=dna_dict_length//threads+1
                
            if dna_dict_length%sample_num==0:
                total_num=dna_dict_length//(sample_num*threads)
            else:
                total_num=dna_dict_length//(sample_num*threads)+1
                
            jobs = []
            for i in range(threads):
                #print str(len(binaryDNAdict_shuf[i*batch:(i+1)*batch]))+" are passed"
                jobs.append(multiprocessing.Process(target=array_saver, 
                                        args=(ooloop, list(range(i*total_num,(i+1)*total_num)), 
                                              [binaryDNAdict[j] for j in shuf2[i*batch:(i+1)*batch]],
                                              [label_list[k] for k in shuf2[i*batch:(i+1)*batch]], 
                                              sample_num, output_dir,)))
            #print("\nsaving train data set to "+output_dir+" with "+str(threads)+" threads")
            for j in jobs:
                j.start()
                
            for j in jobs:
                j.join()
                
            del binaryDNAdict, label_list, jobs
        print("still working on something...")
        with open(output_dir+"data_generation.log", "w") as flog:
            flog.write("Labeled file:"+labeled_genome+"\nClass number:"+str(len(bed_file_list))+"\nExcluded chromosome:"+str(chr_to_skip)+"\n"+to_print1+"\n")
        
        running_time=time.time()-start
        print("Done! A train data set has been saved to "+str(output_dir)+"\nTotal time: "+ str(datetime.timedelta(seconds=running_time)))
    else:
        print("label_file was not created for some reason")
        sys.exit()
if __name__== '__main__':
    main()
    