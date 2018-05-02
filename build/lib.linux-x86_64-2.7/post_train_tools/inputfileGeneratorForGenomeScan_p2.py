import numpy as np
import multiprocessing
import sys
from enhancer_prediction.data_preprocessing_tools.seq_to_binary2 import AGCTtoArray4 as sb4

import psutil
import getopt
       

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
    f=0
    l=0
    _has_seen=set()
    with open(input_file, 'r') as fin:
        seq_list=[]
        position_list=[]
        SEQ=False
        if target_chr=="all":
            for line in fin:
                
                if line.startswith('>'):
                    if not "_" in line and not line.startswith('>chrM'):
                        a=line.split(':')[0]
                        if not a in _has_seen:
                            _has_seen.add(a)
                            print line,
                        position_list.append(line.strip('\n'))
                        SEQ=True
                    else:
                        SEQ=False
                elif SEQ:
                    line=line.strip('\n')
                    data_width=len(line)
                    sequence=sb4(line,data_width)
                    seq_list.append(np.reshape(sequence,[1,1000,4,1] ))
                    f+=1
                    print f
                if f==120000*threads:
                                          
                    jobs = []    
                    for i in range(threads):
                        jobs.append(multiprocessing.Process(target=array_saver, 
                                            args=(str(output_file)+str(i+l*threads), 
                                                  position_list[i*120000:(i+1)*120000], 
                                                  seq_list[i*120000:(i+1)*120000])))
                    for j in jobs:
                        j.start()
                        
                    for j in jobs:
                        j.join()
                    
                    position_list=[]
                    seq_list=[]
                    f=0
                    l+=1
                    
            if len(seq_list)!=0:
                array_saver(str(output_file)+str(l*threads), position_list, seq_list)
                
        
        
        
        
if __name__== '__main__':
    main()
    
    
    
    
    
    
    
    
    
    