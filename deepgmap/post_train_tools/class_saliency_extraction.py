import sys
import gzip
import cPickle
import tensorflow as tf
import numpy as np
import time
import math
import os
import matplotlib.pyplot as plt
import pylab
from deepgmap.post_train_tools import unpooling
import importlib as il
import getopt
from glob import glob 
from natsort import natsorted
import pyBigWig as pbw


"""
conv4_Thu_Jun__7_100401_2018.ckpt-16190.data-00000-of-00001
conv4_Thu_Jun__7_100401_2018.ckpt-16190.index
conv4_Thu_Jun__7_100401_2018.ckpt-16190.meta
conv4_Thu_Jun__7_100401_2018.ckpt-16747.data-00000-of-00001
conv4_Thu_Jun__7_100401_2018.ckpt-16747.index
conv4_Thu_Jun__7_100401_2018.ckpt-16747.meta
conv4_Thu_Jun__7_100401_2018.log
conv4_Thu_Jun__7_100401_2018_plot.pdf
conv4_Thu_Jun__7_100401_2018_trained_variables.npz
conv4_Thu_Jun__7_100401_2018_train_rec.npz
"""


start=time.time()

def genome_scan(filename):
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


BATCH_SIZE=1000
prefix="class_saliency_map"
GPUID="0"
genome_file=""
try:
    options, args =getopt.getopt(sys.argv[1:], 'l:t:o:c:G:g:p:', 
                                 ['log=','test_genome=','output_dir=',"class_of_interest=", "GPUID=", "genoem_file=","prefix="])
except getopt.GetoptError as err:
    print str(err)
    sys.exit(2)
if len(options)<3:
    print('too few argument')
    sys.exit(0)
    
for opt, arg in options:
    if opt in ('-l', '--log'):
        log_file_name=arg
    elif opt in ('-t', '--test_genome'):
        test_genome=arg
    elif opt in ('-o', '--output_dir'):
        output_dir=arg
    elif opt in ('-p','--prefix'):
        prefix=arg
    elif opt in ('-c', "--class_of_interest"):
        class_of_interest=int(arg)
    elif opt in ('-G', "--GPUID"):
        GPUID=arg
    elif opt in ('-g', '--genome_file'):
        genome_file=arg

chromosome_sizes={}
with open(genome_file, "r") as fin:
    for line in fin:
        line=line.split()
        chromosome_sizes[line[0]]=int(line[1])

input_file_prefix= os.path.splitext(log_file_name)[0]
current_variable=np.load(input_file_prefix+"_trained_variables.npz")

with open(log_file_name, 'r') as fin:
    for line in fin:
        if line.startswith('<tf.Variable'):
            line=line.split(' ')
            print line
            if line[1]=="'prediction/W_fc1:0'":
                line=line[2].split('=(')[1].strip(",'")
                first_fc_shape=int(line)
        elif line.startswith("data"):
            line=line.split(':')[1]
            data_length=int(line)
        elif line.startswith("Total class number:"):
            class_num=int(line.split(': ')[1])
test_genome_list=natsorted(glob(test_genome))
if len(test_genome_list)==0:
    sys.exit(test_genome+" does not exist.")

def recon_variable(shape, variable_name):
    initial = tf.truncated_normal(shape, mean=0.02, stddev=0.02)
    return tf.Variable(initial, name=variable_name, trainable=True)
def conv2d_1(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 1, 1], padding='VALID')      

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
def max_pool_4x1(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
def max_pool_2x2_with_argmax(x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
def max_pool_4x1_with_argmax(x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')    


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess2 = tf.Session(config=config)

#x_image_recon = recon_variable([1, 1000, 4, 1], 'x_image_recon')
print GPUID
with tf.device('/device:GPU:'+GPUID):
    x_image_recon=tf.placeholder(tf.float32, shape=[BATCH_SIZE, data_length, 4, 1])
    with tf.GradientTape() as g:
        g.watch(x_image_recon)
        h_conv11_re=conv2d_1(x_image_recon, current_variable["prediction/W_conv1:0"])
        h_conv12_re=conv2d_1(x_image_recon, tf.reverse(current_variable["prediction/W_conv1:0"], [0,1]))
        h_conv11_re_ = tf.nn.relu(h_conv11_re)
        h_conv12_re_ = tf.nn.relu(h_conv12_re)
        h_pool1_re = max_pool_2x2(h_conv11_re_)
        h_pool1_rc_re = max_pool_2x2(h_conv12_re_)
        h_conv2_re = tf.add(tf.nn.relu(conv2d_1(h_pool1_re, current_variable["prediction/W_conv2:0"])), tf.nn.relu(conv2d_1(h_pool1_rc_re, tf.reverse(current_variable["prediction/W_conv2:0"], [0,1]))))
        h_pool2_re = max_pool_2x2(h_conv2_re)
        h_conv21_re = tf.nn.relu(conv2d_1(h_pool2_re, current_variable["prediction/W_conv21:0"]))
        h_pool21_re = max_pool_2x2(h_conv21_re)
        h_conv22_re = tf.nn.relu(conv2d_1(h_pool21_re, current_variable["prediction/W_conv22:0"]))
        h_pool22_re = max_pool_4x1(h_conv22_re)
        
        h_pool3_flat_re = tf.reshape(h_pool22_re, [-1, 1*first_fc_shape])
        
        h_fc1_re = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat_re, current_variable["prediction/W_fc1:0"]), current_variable["prediction/b_fc1:0"]))
        y_conv_re=tf.add(tf.matmul(h_fc1_re,current_variable["prediction/W_fc4:0"]), current_variable["prediction/b_fc4:0"])
        #cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][0])/(tf.nn.sigmoid(y_conv_re[0][2])+tf.nn.sigmoid(y_conv_re[0][0])+tf.nn.sigmoid(y_conv_re[0][1])+0.000001),[1])+tf.reduce_sum(tf.square(x_image_recon))/2000.0
        #print y_conv_re.shape
        #cost =tf.nn.sigmoid(y_conv_re[:,class_of_interest])
        #cost =tf.nn.relu(y_conv_re[:,class_of_interest])
        cost =tf.clip_by_value(y_conv_re[:,class_of_interest], -4.0, 1000000000.0)
        #cost =y_conv_re[:,class_of_interest]
        print cost.shape
    w=g.gradient(cost, x_image_recon)

sess2.run(tf.global_variables_initializer())
#x_image_recon.assign(positive_image[1])
position_list=[]
sal_map=[]
BREAK=False
with pbw.open(output_dir+prefix+".bw", "w") as bw:
    
    chrom_list=[]
    start_list=[]
    end_list=[]
    value_list=[]
    chrom_set=set()
    for test_genome_ in test_genome_list:
        print(test_genome_)
        genome_data=np.load(test_genome_)
        position_list_, seq_list=genome_data['positions'], genome_data['sequences']
        
        
        seq_list=np.array(seq_list, np.int16).reshape(-1, data_length, 4, 1)
        seq_length=seq_list.shape[0]
        print(seq_length)
        
        
        loop=int(math.ceil(float(seq_length)/BATCH_SIZE))
        for i in range(loop):
            if i*BATCH_SIZE>seq_length:
                break
            scanning=seq_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            position=position_list_[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            len_scanning=len(scanning)
            if len_scanning<BATCH_SIZE:
                dummy_array=np.zeros([(BATCH_SIZE-len_scanning), data_length, 4, 1])
                scanning=np.concatenate([scanning, dummy_array])
                
            w_tmp=sess2.run(w, feed_dict={x_image_recon: scanning})
            print w_tmp.shape
            #print w_tmp[1]
            w_tmp_shape=w_tmp.shape
            #print w_tmp[0]
            w_tmp=np.reshape(w_tmp,[w_tmp_shape[0], w_tmp_shape[1],w_tmp_shape[2]])
            #w_tmp=np.amax(np.absolute(np.clip(w_tmp, None, 0.0)), axis=2)
            w_tmp=np.sum(np.absolute(w_tmp), axis=2)
            #w_tmp=np.amax(np.clip(w_tmp, 0.0, None), axis=2)
            #print w_tmp[1]
            #print w_tmp[0]
            #w_tmp=np.amax(w_tmp, axis=2)
            #w_tmp=np.sum(w_tmp, axis=2)
            #print w_tmp[0]
        
            #print w_tmp.shape, len(sal_map)
            #print w_tmp[1:3]
            if len_scanning<BATCH_SIZE:
                w_tmp=w_tmp[:len_scanning]
            for j in range(len_scanning):
                
                sal_map=np.reshape(w_tmp[j], [-1])
                current_chr, current_pos=position[j].strip(">").split(':')
                if not current_chr in chrom_set:
                    chrom_set.add(current_chr)
                    bw.addHeader([(current_chr, chromosome_sizes[current_chr])])
                start, end =map(int, current_pos.split("-"))
                bw.addEntries([current_chr]*len(sal_map), 
                          range(start,end), 
                          ends=range(start+1,end+1), 
                          values=sal_map)
            
            """if len(sal_map)==0:
                sal_map=np.reshape(w_tmp, [-1])
            else:
                sal_map=np.concatenate([sal_map,np.reshape(w_tmp, [-1])],axis=0)
            
            
            if len(sal_map)>=10000000:
                BREAK=True
                break"""
        if BREAK:
            break
sess2.close()
"""
print sal_map.shape
sal_map=sal_map[3050001:]
with open(output_dir+"liver_ctcf_minimum_test.bdg", "w") as fo:
    for i in range(len(sal_map)):
        fo.write("chr2\t"+str(i+3050001)+"\t"+str(i+3050001+1)+"\t"+str(sal_map[i])+"\n")"""
                                               