import numpy as np
import time 
import os.path
import multiprocessing
import sys
import random
#from seq_to_binary import AGCTtoArray2
#import pyximport; pyximport.install()
#import seq_to_binary2
#import seq_to_binary2 as sb2
import importlib as il
sb2=il.import_module("deepgmap.data_preprocessing_tools.seq_to_binary2")
import getopt
import datetime


#convert DNA sequences to a dictionary of onehot vector
def seqtobinarydict(file_, lpos, _chr_to_skip="chr2"):
    lpos=set(lpos)
    binaryDNAdict=[]
    binaryDNAdict_append=binaryDNAdict.append
    position=[]
    position_append=position.append
    seqdata=[]
    s=0
    skip=False
    seqlen=0
    duration=0.0
    i=0
    skipped_chr=''
    for line in file_:
        i+=1
        #start=time.time()
        if line[0]=='>':
            
            if line.startswith(">"+_chr_to_skip+":"):
                if skipped_chr=='':
                    sys.stdout.write("\r" +"skipping "+_chr_to_skip)
                    sys.stdout.flush()
                    skipped_chr=_chr_to_skip
                skip=True
            else:
                
                a=line.strip('>\n')
                if a not in lpos:
                    skip=True
                    if not len(seqdata)==0:
                        binaryDNAdict_append(seqdata)
                        #position_append(a)
                    seqdata=[]
                    continue
                else:
                    skip=False
                    position_append(a)
                    if s%100000==0:
                        sys.stdout.write("\rconverting "+str(a))
                        sys.stdout.flush()
                    
            if not s==0 and not len(seqdata)==0:
                #print(seqdata)
                binaryDNAdict_append(seqdata)
            seqdata=[]
            s+=1
    

        elif not line == '\n' and not line=='' and skip==False: 
            line=line.strip("\n")
            
            seqdata=sb2.AGCTtoArray3(line,len(line))

    if not len(seqdata)==0:
        binaryDNAdict_append(seqdata)
    
    
    return binaryDNAdict, position

def seqtobinarydict2(file_, lpos):
    lpos=set(lpos)
    binaryDNAdict=[]
    binaryDNAdict_append=binaryDNAdict.append
    position=[]
    position_append=position.append
    s=0
    skip=False
    for line in file_:
        if line[0]=='>':
            a=line.strip('>\n')
            if a not in lpos:
                skip=True
            else:
                skip=False
                position_append(a)
                if s%100000==0:
                    sys.stdout.write("\rconverting "+str(a))
                    sys.stdout.flush()
                s+=1

        elif not line == '\n' and not line=='' and skip==False: 
            line=line.strip("\n")
           
            binaryDNAdict_append(sb2.AGCTtoArray3(line.encode('utf-8'),len(line)))
    
    
    return binaryDNAdict, position

def array_saver(ooloop, index_list, binaryDNAdict_shuf,label_list_shuf, sample_num,out_dir):
    #print "binaryDNAdict_shuf length under array_saver: "+str(len(binaryDNAdict_shuf))
    
    for i in range(len(index_list)):
        #print i*sample_num, (i*sample_num+sample_num), index_list[i]
        data_array=np.array(binaryDNAdict_shuf[i*sample_num:(i*sample_num+sample_num)], np.int8)
        #print np.sum(data_array)
        labels=np.array(label_list_shuf[i*sample_num:(i*sample_num+sample_num)], np.int8)
        #print np.shape(labels)
                
        filename = out_dir+"batch_"+str(ooloop)+"_"+str(index_list[i])+".npz"
        #print "saving "+str(filename)
        try:
            with open(filename, "wb") as output_file:
                np.savez_compressed(output_file,labels=labels, data_array=data_array)
        except IOError as e:    
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
        except ValueError:
            print("Could not convert data")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

def array_saver_one_by_one(index, binaryDNAdict_shuf,label_list_shuf,out_dir):

    filename = out_dir+"batch_"+str(index)+".npz"
    try:
        with open(filename, "wb") as output_file:
            np.savez_compressed(output_file,labels=label_list_shuf, data_array=binaryDNAdict_shuf)
    except IOError as e:    
        print("I/O error({0}): {1}".format(e.errno, e.strerror))
    except ValueError:
        print("Could not convert data")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise



def dicttoarray(binaryDNAdict,position, label_list,label_position,reduce_genome):           

    num_seq=len(binaryDNAdict)
    x=0
    y=0

    shuf=range(num_seq)

    random.shuffle(shuf)
    
    binaryDNAdict_shuf=[]
    binaryDNAdict_shuf_append=binaryDNAdict_shuf.append
    label_list_shuf=[]
    label_list_shuf_append=label_list_shuf.append
    k=0
    for i in shuf:
        
        d=binaryDNAdict[i]
        l=label_list[i]           
        dp=position[i]
        lp=label_position[i]
        r=random.random()
        
        #print r, sum(l), reduce_genome
        if sum(l)==0 and r<=reduce_genome:
            #print k
            k+=1
            continue
        else:
            #print dp, lp
            assert dp==lp, "position does not match: %r" %[dp, lp]
            binaryDNAdict_shuf_append(d)
            label_list_shuf_append(l)
            if sum(l)==0:
                x+=1
            else:
                y+=1
        prog=100.0*float(k+y+x)//num_seq
        if prog%10.0==0.0:
            print(prog)
    z=float(x)/float(y+x)
    print(str(k)+" of negative sequences are skipped\n"+"negative/total="+str(z))
    return binaryDNAdict_shuf, label_list_shuf

import gc
def main():      
    start=time.time()
    try:
        options, args =getopt.getopt(sys.argv[1:], 'i:l:o:p:s:r:', ['input_dir=','label=', 'output_dir=','process=','sample_number=','reduce_genome='])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    if len(options)<3:
        print('too few argument')
        sys.exit(0)
        
    sample_num=100
    threads=10
    reduce_genome=60
    
    for opt, arg in options:
        if opt in ('-i', '--input_dir'):
            input_dir=arg
        elif opt in ('-l', '--label'):
            labeled_genome=arg
        elif opt in ('-o', '--output_dir'):
            output_dir=arg
        elif opt in ('-p', '--process'):
            threads=int(arg)
        elif opt in ('-s', '--sample_number'):
            sample_num=int(arg)
        elif opt in ('-r', '--reduce_genome'):
            reduce_genome=float(arg)
            
     
    with open(labeled_genome, 'r') as f2:
        lines=f2.readlines()
        label_position, label_list=sb2.label_reader(lines, "chr2")
    with open(input_dir, 'r') as f1:
        binaryDNAdict, position=seqtobinarydict(f1)
    """
    np_stock=input_dir+".h5"
    if not os.path.exists(np_stock):
        with open(input_dir, 'r') as f1:
            binaryDNAdict, position=seqtobinarydict(f1)
            h5f = h5py.File(input_dir+'.h5', 'w')
            h5f.create_dataset('binaryDNAdict', data=binaryDNAdict, chunks=True)
            h5f.create_dataset('position', data=position, chunks=True)
            h5f.close()
            #np.savez_compressed(input_dir, binaryDNAdict=binaryDNAdict, position=position)
    else:
        np_restore= h5py.File(np_stock,'r')
        binaryDNAdict=np_restore["binaryDNAdict"][:]
        position=np_restore["position"][:]
        np_restore.close()"""
            
    try:        
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as exc:
                if exc.errno != exc.errno.EEXIST:
                    raise
        
        
        binaryDNAdict_shuf, label_list_shuf=dicttoarray(binaryDNAdict,position, label_list,label_position,reduce_genome)
        
        dna_dict_length=len(binaryDNAdict_shuf)
        print(len(label_list_shuf), dna_dict_length)
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
                                    args=(range(i*total_num,(i+1)*total_num), 
                                          binaryDNAdict_shuf[i*batch:(i+1)*batch],
                                          label_list_shuf[i*batch:(i+1)*batch], 
                                          sample_num, output_dir,)))
        
        
        for j in jobs:
            j.start()
            
        for j in jobs:
            j.join()
        print("still working on something...")
    except:
        print("Unexpected error: "+str(sys.exc_info()[0]))
        raise
    gc.collect()
    running_time=time.time()-start
    print("Done!"+"\nTotal time: "+ str(datetime.timedelta(seconds=running_time)))
    
if __name__== '__main__':
    main()
    