import sys
import getopt
import glob as glb
from genome_labeling2 import genome_label
import os
import subprocess as sp
import enhancer_prediction.data_preprocessing_tools.seq_to_binary2 as sb2
from inputfileGenerator_multiple_label3 import seqtobinarydict
from inputfileGenerator_multiple_label3 import dicttoarray
from inputfileGenerator_multiple_label3 import array_saver
import multiprocessing
import time 
import datetime



def run(args):
    main(args)



def main(args=None):
    start=time.time()
    window_size=None
    threads=1
    pref=''
    sample_num=100
    reduce_genome=0.75
    chr_to_skip="chr2"
    if args is not None:
        """
        argparser_bdgpeakcall = subparsers.add_parser( "generate_input",
                                                       help = "Generate a train data set from narrowPeak files." )
        argparser_bdgpeakcall.add_argument( "-b", "--bed", dest = "in_directory", type = str, required = True,
                                            help = "A directory that contains narrowPeak or bed files. \
                                            These files, which indicate signal peak regions in a genome, can be created \
                                            by MACS2 peak caller or other softwares, or can be downloaded from ENCODE project. REQUIRED" )
        argparser_bdgpeakcall.add_argument( "-g", "--genome" , dest = "genome_file_prefix", type = str, required = True,
                                            help = "The directory plus prefix of a bed file and a fasta file that are\
                                            binned with a particular window size. If you have path/to/mm10_1000.bed and\
                                             /path/to/mm10_1000.fa,the input would be '-g /path/to/mm10_1000'. REQUIRED." )
        argparser_bdgpeakcall.add_argument( "-t", "--threads", dest = "thread_number", type = int,
                                           help = "The number of threads. Multithreading is performed only when saving output numpy arrays. Default: 1", default = 1 )
        argparser_bdgpeakcall.add_argument( "-s", "--sample_number", dest = "sample_number", type = int,
                                           help = "The number of samples in a mini-batch. Default: 100", default = 100 )
        argparser_bdgpeakcall.add_argument( "-r", "--reduse_genome", dest="genome_fraction", help="A fraction to ignore signal-negative genome sequences. Default: 0.75",default=0.75)
        argparser_bdgpeakcall.add_argument("-p", "--prefix", dest="out_prefix",
                                            help = "The prefix of output files and folders. Default: ''", default = '' )
        """    
        bed_file_dir=args.in_directory
        genome_pref=args.genome_file_prefix
        threads=args.thread_number
        sample_num=args.sample_number
        reduce_genome=args.genome_fraction
        pref=args.out_prefix
        chr_to_skip=args.chromosome_to_skip
    else:
            
        try:
            options, args =getopt.getopt(sys.argv[1:], 'b:g:w:t:p:s:r:h:', ['bed=', 'genome=', 'window_size=' 'threads=', 'prefix=','sample_number=','reduce_genome=','help='])
        except getopt.GetoptError as err:
            print str(err)
            sys.exit(2)
    
        
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
                
        """
        bedtools intersect -F 0.4 -f 0.9 -e -u -a /home/fast/onimaru/data/genome_fasta/mm10_1000.bed -b $i >./$i\_1000.bed
        """ 
    genome_1000=genome_pref+".bed"
    genome_fasta=genome_pref+".fa"
    if os.path.isfile(genome_1000)==False or os.path.isfile(genome_fasta)==False:
        print "can't find genome files"
        sys.exit()
    with open(genome_1000, 'r') as fin:
        line=fin.readline()
        line=line.split()
        window_size=int(line[2])-int(line[1])
    bed_file_list=[]
    if bed_file_dir.endswith('.narrowPeak'):
        bed_file_list.append(bed_file_dir)
        out_dir=bed_file_dir
    else:
        if not bed_file_dir.endswith("/"):
            bed_file_dir+="/"
        bed_file_dir_=bed_file_dir+"*.narrowPeak"
        bed_file_list=glb.glob(bed_file_dir_)
        out_dir=bed_file_dir
    print("reading narrowPeak files of "+ str(bed_file_list))
    if len(bed_file_list)==0:
        print("No peak files in "+str(bed_file_dir))
        print(howto)
        sys.exit()
    bed_file_list_2=[]
    for b in bed_file_list:
        b_=b.split(".")[0]+"_"+str(window_size)+".bed"
        tmp_out=open(b_, 'w')
        try:
            sp.check_call(["bedtools", "intersect","-F", "0.4", "-f", "0.9", "-e", "-u", "-a", str(genome_1000), "-b", str(b)], stdout=tmp_out)
        except OSError as e:
            if e.errno == os.errno.ENOENT:
                print str(b)+" not found"
            else:
                print "Something went wrong while trying to run bedtools"
            raise
        tmp_out.close()
        bed_file_list_2.append(b_)
        
    genome_label(bed_file_list_2, genome_1000,bed_file_dir,pref)
    head, tail = os.path.split(genome_1000)
    labeled_genome=str(out_dir)+str(pref)+'_'+str(tail)+'.labeled'
    output_dir=str(out_dir)+str(pref)+'_'+str(tail.split('.')[0])+"s"+str(sample_num)+"r"+str(reduce_genome)+'_train_data_set/'
    print("outputting train data set to "+output_dir)
    if os.path.isfile(labeled_genome):
        
        with open(labeled_genome, 'r') as f2:
            label_position, label_list=sb2.label_reader(f2, chr_to_skip)
        with open(genome_fasta, 'r') as f1:
            binaryDNAdict, position=seqtobinarydict(f1, chr_to_skip)

                
        try:        
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except OSError as exc:
                    if exc.errno != exc.errno.EEXIST:
                        raise
            binaryDNAdict_shuf, label_list_shuf=dicttoarray(binaryDNAdict,position, label_list,label_position,reduce_genome)
    
            dna_dict_length=len(binaryDNAdict_shuf)
    
            if dna_dict_length%threads==0:
                batch=dna_dict_length/threads
            else:
                batch=dna_dict_length/threads+1
                
            if dna_dict_length%sample_num==0:
                total_num=dna_dict_length/(sample_num*threads)
            else:
                total_num=dna_dict_length/(sample_num*threads)+1
                
            jobs = []
            for i in range(threads):
                #print str(len(binaryDNAdict_shuf[i*batch:(i+1)*batch]))+" are passed"
                jobs.append(multiprocessing.Process(target=array_saver, 
                                        args=(range(i*total_num,(i+1)*total_num), 
                                              binaryDNAdict_shuf[i*batch:(i+1)*batch],
                                              label_list_shuf[i*batch:(i+1)*batch], 
                                              sample_num, output_dir,)))
            print("saving data set with "+str(threads)+" threads")
            for j in jobs:
                j.start()
                
            for j in jobs:
                j.join()
            print("still working on something...")
        except:
            print("Unexpected error: "+str(sys.exc_info()[0]))
            raise
        
        running_time=time.time()-start
        print "Done!"+"\nTotal time: "+ str(datetime.timedelta(seconds=running_time))
    else:
        print("label_file was not created for some reason")
        sys.exit()
if __name__== '__main__':
    main()
    