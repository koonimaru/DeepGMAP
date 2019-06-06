import sys
import gzip
import cPickle
import tensorflow as tf
import numpy as np
import time
import math
import os
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
import pylab
from deepgmap.post_train_tools import unpooling
import importlib as il
import getopt
import deepgmap.post_train_tools.sequence_visualizer2 as sv2


def test_batch(test_batch_file, _batch_size,_data_length):
    with np.load(test_batch_file) as f:
        dnase_data_labels1=f['labels'], f['data_array']
        images=np.reshape(dnase_data_labels1[1], (_batch_size, _data_length, 4, 1))
        labels=dnase_data_labels1[0]
    return images, labels


def run():
    start=time.time()
    
    
    try:
        options, args =getopt.getopt(sys.argv[1:], 'm:t:n:o:d:', ['model=','test_batch=','network_constructor=','output_dir=','deconv='])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    if len(options)<3:
        print('too few argument')
        sys.exit(0)
    for opt, arg in options:
        if opt in ('-m', '--model'):
            trained_model=arg
        elif opt in ('-t', '--test_batch'):
            test_batch_file=arg
        elif opt in ('-n', '--network_constructor'):
            network_constructor=arg
        elif opt in ('-o', '--output_dir'):
            output_dir=arg
        elif opt in ('-d','--deconv'):
            deconv=arg
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if not output_dir.endswith("/"):
        output_dir+="/"
    #output_dir=None
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    keep_prob = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    keep_prob3 = tf.placeholder(tf.float32)
    
    
    x_image = tf.placeholder(tf.float32, shape=[None, 1000, 4, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 12])
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
                     output_dir=output_dir+start_at,
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

    def recon_variable(shape, variable_name):
        initial = tf.truncated_normal(shape, mean=0.02, stddev=0.01)
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
        
    x_image_recon = recon_variable([1, 1000, 4, 1], 'x_image_recon')
    
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
    print(y_conv_re.shape)
    #sys.exit()
    #cost =-tf.reshape(y_conv_re,[1])+tf.reduce_sum(tf.square(x_image_recon))/500.0
    #cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][1])-tf.nn.sigmoid(y_conv_re[0][0]),[1])+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    #cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][1])-tf.nn.sigmoid(y_conv_re[0][0]),[1])*tf.log(tf.nn.sigmoid(y_conv_re[0][1]+0.000001)/(tf.nn.sigmoid(y_conv_re[0][0])+0.000001))+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    #cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][1])*tf.nn.sigmoid(y_conv_re[0][0]),[1])+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    #cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][0])/(tf.nn.sigmoid(y_conv_re[0][2])+tf.nn.sigmoid(y_conv_re[0][1])+0.000001),[1])+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][0])/(tf.nn.sigmoid(y_conv_re[0][2])+tf.nn.sigmoid(y_conv_re[0][1])+0.000001),[1])+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    #cost =-tf.reshape(tf.reduce_sum(tf.nn.sigmoid(y_conv_re[0][3:6]))/(tf.reduce_sum(tf.nn.sigmoid(y_conv_re[0][0:3]))\
                                                                       #+tf.reduce_sum(tf.nn.sigmoid(y_conv_re[0][6:]))+0.000001),[1])\
                                                                       #+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    
    #cost =-tf.reshape(tf.reduce_sum(tf.nn.sigmoid(y_conv_re[0][6:12]))/(tf.reduce_sum(tf.nn.sigmoid(y_conv_re[0][0:12]))\
                                                                       #+0.000001),[1])\
                                                                       #+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    #cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][2])+tf.nn.sigmoid(y_conv_re[0][0])+tf.nn.sigmoid(y_conv_re[0][1]),[1])+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    
    
    #cost =-y_conv_re+tf.reduce_sum(tf.square(x_image_recon))/500.0
    #cost =-(y_conv_re[0][1]-y_conv_re[0][0])+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    
    #cost =-tf.reshape(y_conv_re,[1])
    train_step2 = tf.train.AdamOptimizer(0.001).minimize(cost)

    sess2.run(tf.global_variables_initializer())
    #x_image_recon.assign(positive_image[1])
    cost_list=[]
    for i in range(100000):
        _, progress, y_val, a, b  =sess2.run([train_step2, cost, y_conv_re, h_conv11_re_, h_conv22_re])
        cost_list.append(progress)
        if (i+1)%100==0:
            print('step '+str(i)+' cost: '+str(progress)) #+', y_conv_re: '+str(y_val)
        #if progress<=-0.9999:
            #break
        if i>20000:
            if cost_list[i-500]-cost_list[i]<=0.0000001:
                print(cost_list[i-100]-cost_list[i])
                break
        #print str(a.shape)+'\n'+str(b.shape)+'\n'+str(1*fc1_param*dimension3)
 
        
            #print a
            
    output_dir=output_dir+"/"+os.path.split(trained_model)[1]+str(start_at)+"_ese14_re"
    final_recon=x_image_recon.eval(session=sess2)
    final_recon_res=np.reshape(final_recon, (data_length, 4))
    inputdir=input_dir.split('/')[-1]
    np.savez_compressed(output_dir, recon=final_recon_res)
    sess2.close()
    
    
    fig = plt.figure(figsize=(8,8))
        # Plot distance matrix.
    axmatrix = fig.add_axes([0.1,0.05,0.6,0.9])
    im = axmatrix.matshow(final_recon_res, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    axcolor = fig.add_axes([0.8,0.05,0.02,0.9])
    pylab.colorbar(im, cax=axcolor)
    fig.savefig(output_dir+'.png')
    import deepgmap.post_train_tools.sequence_visualizer2 as sq
    sq.seuquence_visualizer2(final_recon_res, output_dir+'.pdf')
    
    sv2.seuquence_visualizer2(final_recon_res, output_dir+'_motif.pdf')
        
if __name__== '__main__':
    run()