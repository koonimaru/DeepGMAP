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
import data_preprocessing_tools.seq_to_binary2 as sb2

def DNA_to_array_converter(input_file,target_chr="chr2"):
    seq_list=[]
    position_list=[]
    
    
    with open(input_file, 'r') as fin:
    
        SEQ=False
        if target_chr=="all":
            for line in fin:
                if line[:4]=='>chr':
                    if not "_" in line and not line.startswith('>chrM'):
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
        else:
            for line in fin:
                if line[:4]=='>chr':
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

def array_saver(index_list, position_list,seq_list, sample_num,out_dir):
    #print "binaryDNAdict_shuf length under array_saver: "+str(len(binaryDNAdict_shuf))
    
    for i in range(len(index_list)):
        data_array=np.array(position_list[i*sample_num:(i*sample_num+sample_num)], np.int32)
        #print np.sum(data_array)
        labels=seq_list[i*sample_num:(i*sample_num+sample_num)]
        #print np.shape(labels)
                
        filename = out_dir+"_"+str(index_list[i])+".npz"
        #print "saving "+str(filename)
        if not os.path.exists(os.path.dirname(filename)):
            try:
                os.makedirs(os.path.dirname(filename))
            except OSError as exc:
                if exc.errno != exc.errno.EEXIST:
                    raise
        try:
            with open(filename, "wb") as output_file:
                np.savez_compressed(output_file,labels=labels, data_array=data_array)
        except IOError as e:    
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
        except ValueError:
            print "Could not convert data"
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise

def main():
    input_file, target_chr, output_file=sys.argv[1:]
    position_list, seq_list=DNA_to_array_converter(input_file,target_chr=target_chr)
    seq_num=len(position_list)
    print seq_num
    if target_chr=="all":
        DIVIDES_NUM=23
    else:
        DIVIDES_NUM=3
        
    chunk_num=seq_num/DIVIDES_NUM+1
    for i in range(DIVIDES_NUM):
        if not i*chunk_num>seq_num:
            np.savez_compressed(str(output_file)+str(i), positions=position_list[i*chunk_num:(i+1)*chunk_num], sequences=seq_list[i*chunk_num:(i+1)*chunk_num])    
            print (i+1)*chunk_num
if __name__== '__main__':
    main()
    
    
    
    
    
    
    