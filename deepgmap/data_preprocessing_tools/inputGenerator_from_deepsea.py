import numpy as np
import sys
import random
import multiprocessing
import os

def dicttoarray(binaryDNAdict, label_list):           

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
        #dp=position[i]
        #lp=label_position[i]
        r=random.random()
        
        #print r, sum(l), reduce_genome
            #print dp, lp
            #assert dp==lp
        binaryDNAdict_shuf_append(d)
        label_list_shuf_append(l)
        if sum(l)==0:
            x+=1
        else:
            y+=1
        prog=100.0*float(k+y+x)/num_seq
        if prog%10.0==0.0:
            print(str(prog)+" of data are shuffled.")
    z=float(x)/float(y+x)
    print(str(k)+" of negative sequences are skipped\n"+"negative/total="+str(z))
    return binaryDNAdict_shuf, label_list_shuf


def array_saver(index_list, binaryDNAdict_shuf,label_list_shuf, sample_num,out_dir):
    #print "binaryDNAdict_shuf length under array_saver: "+str(len(binaryDNAdict_shuf))
    
    for i in range(len(index_list)):
        data_array=np.array(binaryDNAdict_shuf[i*sample_num:(i*sample_num+sample_num)], np.int32)
        #print np.sum(data_array)
        labels=np.array(label_list_shuf[i*sample_num:(i*sample_num+sample_num)], np.int32)
        #print np.shape(labels)
                
        filename = out_dir+"batch_"+str(index_list[i])+".npz"
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

fname="/home/fast/onimaru/encode/deepsea/deepsea_train/train.npz"
output_dir=os.path.split(fname)[0]+"/train_data_for_my_program/"
os.makedirs(output_dir)
fload=np.load(fname)
data=fload["data_array"]
labels=fload["labels"]

binaryDNAdict_shuf, label_list_shuf=dicttoarray(data, labels,)

dna_dict_length=len(binaryDNAdict_shuf)

if dna_dict_length%16==0:
    batch=dna_dict_length/16
else:
    batch=dna_dict_length/16+1
    
if dna_dict_length%100==0:
    total_num=dna_dict_length/(100*16)
else:
    total_num=dna_dict_length/(100*16)+1
    
jobs = []
for i in range(16):
    #print str(len(binaryDNAdict_shuf[i*batch:(i+1)*batch]))+" are passed"
    jobs.append(multiprocessing.Process(target=array_saver, 
                            args=(range(i*total_num,(i+1)*total_num), 
                                  binaryDNAdict_shuf[i*batch:(i+1)*batch],
                                  label_list_shuf[i*batch:(i+1)*batch], 
                                  100, output_dir,)))
print("saving data set with "+str(16)+" threads")
for j in jobs:
    j.start()
    
for j in jobs:
    j.join()














