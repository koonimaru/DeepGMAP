
import sys
import gzip
import cPickle
import tensorflow as tf
import numpy as np
import time
import math
import os
from natsort import natsorted, ns
import network_constructor_multiple_label as nc
import subprocess as sp
start=time.time()
#dimension1_2=16

#with gzip.open('/media/koh/HD-PCFU3/mouse/filter1_999_Tue_Oct_25_122720_2016.cpickle.gz', 'r') as f:
 #   saved_variables=cPickle.load(f)
 #   W_conv1, W_conv2, W_conv3, b_conv1, b_conv2, b_conv3, W_fc1, W_fc2, W_fc3, W_fc4, b_fc1, b_fc2, b_fc3, b_fc4=saved_variables

import glob
def genome_scan(filename):
 #/media/koh/HD-PCFU3/mouse/test_genome/genome_chr1_06_250plus.cpickle.gz
    with open(filename, 'r') as f1:
        file_name=f1.name
        path_sep=os.path.sep
        file_name1=file_name.split(path_sep)
        file_name2=file_name1[-1].split('_')
        chromosome=file_name2[2]
        a=file_name2[3]
        b=a.split('.')
        chr_position=int(b[0])
        #window_id=(file_name2[3])[:3]
        genome_seq=np.load(f1)
        shape_of_genome=genome_seq['genome'].shape
        genome_seq_re=np.reshape(genome_seq['genome'], (shape_of_genome[0], shape_of_genome[1], 4, 1))
        genome_seq_re_list=np.array_split(genome_seq_re, 100)
    return genome_seq_re_list, chromosome, chr_position #, window_id
    
def process(f, out_dir):
    sess = tf.Session()
    x_image = tf.placeholder(tf.float32, shape=[None, 1000, 4, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 19])
    keep_prob = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    keep_prob3 = tf.placeholder(tf.float32)
    phase=tf.placeholder(tf.bool)
    data_length=1000
    if 'ckpt' in sys.argv[1].rsplit('.', 1)[1]: 
        input_dir=sys.argv[1]
    elif 'meta'  in sys.argv[1].rsplit('.', 1)[1] or 'index'  in sys.argv[1].rsplit('.', 1)[1]:
        input_dir=sys.argv[1].rsplit('.', 1)[0]
    else:
        print("the input file should be a ckpt file")
        sys.exit(1)
    
    model = nc.Model(image=x_image, label_dim=19, label=y_, phase=phase, output_dir=None, start_at=None, keep_prob=keep_prob, keep_prob2=keep_prob2, keep_prob3=keep_prob3, data_length=data_length)
    sess.run(tf.global_variables_initializer())
    saver=model.saver
    try:
        saver.restore(sess, input_dir)
    except:
        print("can't open "+str(input_dir))
        sys.exit(0)
    for seq in f:
        
        try:
            genome_seq_re_list, chromosome, chr_position=genome_scan(seq)
        except:
            print("can't open "+str(file[0]))
            sys.exit(0)
        y_prediction1=[]
        i=0
        for i in range(len(genome_seq_re_list)):
            scanning=genome_seq_re_list[i]
            y_prediction2 =np.array(sess.run(model.prediction[0], feed_dict={x_image: scanning, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False}), np.float64)
            y_prediction1.append(1.0 / (1.0 + np.exp(-y_prediction2)))
            if i%10==0:
                print('scanning '+str(chromosome)+'_'+str(chr_position)+', '+str(100*i/len(genome_seq_re_list))+' %')    
        filename_1=str(out_dir)+str(chromosome)+'.bed'
        print('writing '+filename_1)
        if os.path.isfile(filename_1):
            output_handle=open(filename_1, 'a')
        else:
            output_handle=open(filename_1, 'w')
        i=0
        j=0
        y_len=len(y_prediction1)
        for j in range(y_len):
            y_len_j=len(y_prediction1[j])
            for i in range(y_len_j):
                value=np.max(y_prediction1[j][i][:-1])-y_prediction1[j][i][-1]
                if value>0.0:
                    if int(sys.argv[4])==500:
                        start_pos=int(chr_position)*int(1e7)+500*i+200*500*j
                        end_pos=start_pos+499
                    elif int(sys.argv[4])==1000:
                        start_pos=int(chr_position)*int(1e7)+1000*i+100*1000*j
                        end_pos=start_pos+999
                
                    output_handle.write(str(chromosome)+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t'+str(value)+'\n')
                    if i%10==0:
                        print(str(str(chromosome)+'\t'+str(start_pos)+'\t'+str(end_pos)+'\t'+str(value))) 
        output_handle.close()
        print('finished writing '+filename_1)
    sess.close()
    out=open(str(out_dir)+str(chromosome)+"_srt.bed", 'w')
    sp.check_call(["bedtools", "sort","-i", str(filename_1)], stdout=out)
    out.close()

import multiprocessing    
def main():
    input_dir=sys.argv[1].rsplit('.', 1)[0]
    
        
    path_sep=os.sep
    file_name=input_dir.split(path_sep)
    a=time.asctime()
    b=a.replace(':', '')
    start_at=b.replace(' ', '_')
    out_dir=sys.argv[2]+file_name[-1]
    
    if not os.path.exists(os.path.dirname(out_dir)):
            try:
                os.makedirs(os.path.dirname(out_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != exc.errno.EEXIST:
                    raise

    start=time.time()
    s=0
    try:
        f = glob.glob(sys.argv[3])
        process(f, out_dir)
        #x=p.apply_async(process, (t_,))
        #x.get()
    except :
        print("Unexpected error:", sys.exc_info()[0])
        raise      
        
    #for i in f:
    #  process(i, out_dir)
    
      
        
    print(time.time()-start)


if __name__== '__main__':
    main()



