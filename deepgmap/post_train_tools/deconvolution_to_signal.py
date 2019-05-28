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
from natsort import natsorted
from glob import glob
def test_batch(test_batch_file):
    with np.load(test_batch_file) as f:
        dnase_data_labels1=f['labels'], f['data_array']
        images=np.reshape(dnase_data_labels1[1], (batch_size, data_length, 4, 1))
        labels=dnase_data_labels1[0]
    return images, labels

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
start=time.time()
try:
    options, args =getopt.getopt(sys.argv[1:], 'm:t:n:o:', ['model=','test_genome=','network_constructor=','output_dir='])
except getopt.GetoptError as err:
    #print str(err)
    sys.exit(2)
if len(options)<3:
    print('too few argument')
    sys.exit(0)
for opt, arg in options:
    if opt in ('-m', '--model'):
        trained_model=arg
    elif opt in ('-t', '--test_genome'):
        test_genome=arg
    elif opt in ('-n', '--network_constructor'):
        network_constructor=arg
    elif opt in ('-o', '--output_dir'):
        output_dir=arg


#output_dir=None

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

keep_prob = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)


x_image = tf.placeholder(tf.float32, shape=[None, 1000, 4, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 3])
phase=tf.placeholder(tf.bool)
dropout_1=0.95
dropout_2=0.9
dropout_3=0.85
batch_size=100
data_length=1000 
input_dir=trained_model
nc=il.import_module("deepgmap.network_constructors."+str(network_constructor))   
train_speed=0.00005
a=time.asctime()
b=a.replace(':', '')
start_at=b.replace(' ', '_')

model = nc.Model(image=x_image, label=y_, 
                 output_dir=output_dir,
                 phase=phase, 
                 start_at=start_at, 
                 keep_prob=keep_prob, 
                 keep_prob2=keep_prob2, 
                 keep_prob3=keep_prob3, 
                 data_length=data_length,
                 max_to_keep=2,
                 GPUID="1")


sess.run(tf.global_variables_initializer())
saver=model.saver
saver.restore(sess, input_dir)

test_genome_list=natsorted(glob(test_genome))
if len(test_genome_list)==0:
    sys.exit(test_genome+" does not exist.")
 
def conv2d_tp(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 1, 1, 1], padding='VALID')
def conv2d_tp2(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1,2, 1,1], padding='VALID') 
def conv2d_tp4(x, W, output_shape):
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 4, 1, 1], padding='VALID')    
def max_pool_2x1(x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
def max_pool_4x1(x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
position_list=[]
y_prediction2=[]
for test_genome_ in test_genome_list:
    print(test_genome_)
    genome_data=np.load(test_genome_)
    position_list_, seq_list=genome_data['positions'], genome_data['sequences']
    if len(position_list)==0:
        position_list=position_list_
    else:
        position_list=np.concatenate([position_list,position_list_])
    seq_list=np.array(seq_list, np.int16).reshape(-1, data_length, 4, 1)
    seq_length=seq_list.shape[0]
    print(seq_length)
    
    
    loop=int(math.ceil(float(seq_length)/BATCH_SIZE))
    for i in range(loop):
        if i*BATCH_SIZE>seq_length:
            break
        scanning=seq_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        if len(y_prediction2)==0:
            _, y_prediction2,variavl_dict, neurons_dict,_2=sess.run(model.prediction, feed_dict={x_image: scanning, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
            
        else:
            _, y_prediction1,variavl_dict, neurons_dict,_2=sess.run(model.prediction, feed_dict={x_image: scanning, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
            y_prediction2=np.concatenate([y_prediction2,y_prediction1],axis=0)

        
        h_conv11_,\
        h_conv12_,\
        h_conv2_,\
        h_conv21_,\
        h_conv22_,\
        h_pool1_,\
        h_pool1_rc_,\
        h_pool2_,\
        h_pool21_,\
        h_pool22_ =\
        neurons_dict["h_conv11"],\
        neurons_dict["h_conv12"],\
        neurons_dict["h_conv2"],\
        neurons_dict["h_conv21"],\
        neurons_dict["h_conv22"],\
        neurons_dict["h_pool1"],\
        neurons_dict["h_pool1_rc"],\
        neurons_dict["h_pool2"],\
        neurons_dict["h_pool21"],\
        neurons_dict["h_pool22"]
    
        
        sess2 = tf.Session()
        #print h_pool21_
        h_pool21_shape=list(h_pool21_.shape)
        h_pool21_t4= conv2d_tp(h_conv22_, variavl_dict["W_conv22"], h_pool21_shape)
        _, mask21=max_pool_2x1(h_conv21_)
        #h_unpool21_t4=unpooling.unpool(h_pool21_t4, mask21,output_shape=h_conv21_.shape)
        h_unpool21_t4=unpooling.unpool2(h_pool21_t4, mask21)
        
        h_pool2_shape=list(h_pool2_.shape)
        h_pool2_t4= conv2d_tp(h_unpool21_t4, variavl_dict["W_conv21"], h_pool2_shape)
        _, mask2=max_pool_2x1(h_conv2_)
        #h_unpool2_t4=unpooling.unpool(h_pool2_t4,mask2,output_shape=h_conv2_.shape)
        h_unpool2_t4=unpooling.unpool2(h_pool2_t4,mask2)
        
        h_pool1_shape=list(h_pool1_.shape)
        h_pool1_t4= conv2d_tp(h_unpool2_t4, variavl_dict["W_conv2"], h_pool1_shape)
        _,mask1=max_pool_2x1(h_conv11_)
        #h_unpool1_t4=unpooling.unpool(h_pool1_t4,mask1,output_shape=h_conv11_.shape)
        h_unpool1_t4=unpooling.unpool2(h_pool1_t4,mask1)
        
        h_pool1_rc_t4=conv2d_tp(h_unpool2_t4, tf.reverse(variavl_dict["W_conv2"], [0,1]), h_pool1_shape)
        _,mask1rc=max_pool_2x1(h_conv12_)
        #h_unpool1_rc_t4=unpooling.unpool(h_pool1_rc_t4,mask1rc,output_shape=h_conv12_.shape)
        h_unpool1_rc_t4=unpooling.unpool2(h_pool1_rc_t4,mask1rc)
        
        reconstruction_shape=scanning.shape
        #print reconstruction_shape
        reconstruction_conv22=conv2d_tp(h_unpool1_t4, variavl_dict["W_conv1"], reconstruction_shape)+conv2d_tp(h_unpool1_rc_t4, tf.reverse(variavl_dict["W_conv1"], [0,1]), reconstruction_shape)
        
        
        sess2.run(tf.global_variables_initializer())
        units_conv22 = sess2.run(reconstruction_conv22)
        reshaped_conv22=np.reshape(units_conv22, (data_length, 4))
        
        sess2.close()
        
        
sess.close()
                                               