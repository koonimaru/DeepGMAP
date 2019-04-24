import numpy as np
import os.path
import multiprocessing
import sys
import deepgmap.data_preprocessing_tools.seq_to_binary2 as sb2
import psutil
import getopt
import time


def div_roundup(x, y):
    if y%x==0:
        return y//x
    else:
        return y//x+1


def DNA_to_array_converter(input_file_read,seq_num,target_chr):
    seq_list=[]
    seq_list_append=seq_list.append
    position_list=[]
    position_list_append=position_list.append
    b1=0.0
    i=0
    
    data_width=len(input_file_read[1].strip("\n"))
    print(data_width)
    SEQ=False
    #print seq_list
    for l, line in enumerate(input_file_read):
        if line.startswith('>'):
            #if not "_" in line and not line.startswith('>chrM'):
            if not line.startswith('>chrM'):
                #print line,
                position_list_append(line.strip('\n'))
                SEQ=True
            else:
                SEQ=False
            if i%100000==0:
                print(line)
        elif SEQ:
            line=line.strip('\n')
            
            #a1=time.time()
            seq_list_append(sb2.AGCTtoArray4(line,data_width))
            
            #b1+=time.time()-a1
        i+=1
        #if i%100000==0:
            #print b1
            #sys.exit()
                    
    return position_list, seq_list
        

def array_saver(outfile,positions,sequences):
    print('saving '+outfile)
    np.savez_compressed(outfile,positions=positions,sequences=sequences)
        
def run(args):
    
    main(args)

def main(args=None):
    if args is not None:
        input_file=args.input_genome
        target_chr=args.chromosome
        output_file=args.out_directory
        threads=args.thread_number
        chunck_data=args.chunck_data
        print(args)
    else:
        try:
            options, args =getopt.getopt(sys.argv[1:], 'i:t:o:p:', ['input_dir=','target_chr=', 'output_dir=','process='])
        except getopt.GetoptError as err:
            print(str(err))
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
    
        print(options)
    file_size=os.path.getsize(input_file)
    print(file_size)
    
    loop_to_reduce_ram=div_roundup(1000000000, file_size)
    try:
        with open(input_file, "r") as fin:
            input_file_read=fin.readlines()
    except IOError:
        print('cannot open', input_file)
    
    line_num=len(input_file_read)
    #print line_num
    seq_num=line_num/2
    
    sub_seq_num=div_roundup(loop_to_reduce_ram, seq_num)    
    DIVIDES_NUM=div_roundup(120000, sub_seq_num)
    
    for l1 in range(loop_to_reduce_ram):
    
        position_list, seq_list=DNA_to_array_converter(input_file_read[2*l1*sub_seq_num:2*(l1+1)*sub_seq_num],sub_seq_num,target_chr)
        
        print(position_list[0], input_file_read[2*l1*sub_seq_num])
    
    
        outerloop=div_roundup(threads, DIVIDES_NUM)
        chunk_num=div_roundup(DIVIDES_NUM, sub_seq_num)            
            
        if DIVIDES_NUM>=threads:
            job_num=threads
        else:
            job_num=DIVIDES_NUM
            
        print(DIVIDES_NUM, threads, outerloop, job_num)
        
        
        for l in range(outerloop):
            jobs = []    
            for i in range(job_num):
                if i*chunk_num+l*job_num*chunk_num>sub_seq_num:
                    break
                jobs.append(multiprocessing.Process(target=array_saver, 
                                    args=(str(output_file)+"_"+str(l1)+"_"+str(i+l*job_num), 
                                          position_list[i*chunk_num+l*job_num*chunk_num:(i+1)*chunk_num+l*job_num*chunk_num], 
                                          seq_list[i*chunk_num+l*job_num*chunk_num:(i+1)*chunk_num+l*job_num*chunk_num])))
            for j in jobs:
                j.start()
                
            for j in jobs:
                j.join()
        
        
    
if __name__== '__main__':
    main()
    
    
    
    
    
    
    
    
    
    