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
def test_batch(test_batch_file):
    with np.load(test_batch_file) as f:
        dnase_data_labels1=f['labels'], f['data_array']
        images=np.reshape(dnase_data_labels1[1], (batch_size, data_length, 4, 1))
        labels=dnase_data_labels1[0]
    return images, labels

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
try:
    options, args =getopt.getopt(sys.argv[1:], 'm:t:n:o:', ['model=','test_genome=','network_constructor=','output_dir='])
except getopt.GetoptError as err:
    print str(err)
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
        
test_genome_list=natsorted(glob(test_genome))
if len(test_genome_list)==0:
    sys.exit(test_genome+" does not exist.")

#output_dir=None

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

keep_prob = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)


x_image = tf.placeholder(tf.float32, shape=[None, 1000, 4, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 54])
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


if 'ckpt' in input_dir.rsplit('.', 1)[1]: 
    input_dir=input_dir
elif 'meta'  in input_dir.rsplit('.', 1)[1] or 'index'  in input_dir.rsplit('.', 1)[1]:
    input_dir=input_dir.rsplit('.', 1)[0]
else:
    print("the input file should be a ckpt file")
    sys.exit(1)
sess.run(tf.global_variables_initializer())
saver=model.saver
saver.restore(sess, input_dir)

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



index_of_image=0
positive_image=[]
"""for y in batch[1]:
    print y[0]
    if y[0]==1:
        positive_image.append(np.reshape(batch[0][index_of_image], [1,1000,4,1]))
    index_of_image+=1"""
    
#print positive_image[0]
current_variable={}
all_tv=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
for v in all_tv:
    value=sess.run(v)
    scope=v.name
    current_variable[scope]=value
fc1_param=model.fc1_param
dimension22=model.dimension22

sess.close()

sess2 = tf.Session()
    
#x_image_recon = recon_variable([1, 1000, 4, 1], 'x_image_recon')
x_image_recon=tf.placeholder(tf.float32, shape=[None, 1000, 4, 1])
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
    
    h_pool3_flat_re = tf.reshape(h_pool22_re, [-1, 1*fc1_param*dimension22])
    
    h_fc1_re = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat_re, current_variable["prediction/W_fc1:0"]), current_variable["prediction/b_fc1:0"]))
    y_conv_re=tf.add(tf.matmul(h_fc1_re,current_variable["prediction/W_fc4:0"]), current_variable["prediction/b_fc4:0"])
    #cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][0])/(tf.nn.sigmoid(y_conv_re[0][2])+tf.nn.sigmoid(y_conv_re[0][0])+tf.nn.sigmoid(y_conv_re[0][1])+0.000001),[1])+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    #print y_conv_re.shape
    cost =tf.nn.sigmoid(y_conv_re[:,39])
    print cost.shape
w=g.gradient(cost, x_image_recon)

sess2.run(tf.global_variables_initializer())
#x_image_recon.assign(positive_image[1])
position_list=[]
sal_map=[]
BREAK=False
with pbw.open(output_dir+"liver_ctcf_minimum_test.bw", "w") as bw:
    bw.addHeader([("chr2", 182113000)])
    chrom_list=[]
    start_list=[]
    end_list=[]
    value_list=[]
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
            
            w_tmp=sess2.run(w, feed_dict={x_image_recon: scanning})
            print w_tmp.shape
            #print w_tmp[1]
            w_tmp_shape=w_tmp.shape
            #print w_tmp[0]
            #w_tmp=np.absolute(np.reshape(w_tmp,[w_tmp_shape[0], w_tmp_shape[1],w_tmp_shape[2]]))
            w_tmp=np.absolute(np.clip(np.reshape(w_tmp,[w_tmp_shape[0], w_tmp_shape[1],w_tmp_shape[2]]), None, 0.0))
            #print w_tmp[1]
            #print w_tmp[0]
            w_tmp=np.amax(w_tmp, axis=2)
            #print w_tmp[0]
        
            #print w_tmp.shape, len(sal_map)
            #print w_tmp[1:3]
            sal_map=np.reshape(w_tmp, [-1])
            
            bw.addEntries(["chr2"]*len(sal_map), 
                          range(i*BATCH_SIZE*data_length,(i+1)*BATCH_SIZE*data_length), 
                          ends=range(i*BATCH_SIZE*data_length+1,(i+1)*BATCH_SIZE*data_length+1), 
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
                                               