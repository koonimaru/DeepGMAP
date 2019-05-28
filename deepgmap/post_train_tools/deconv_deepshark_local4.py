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


def test_batch(test_batch_file):
    with np.load(test_batch_file) as f:
        dnase_data_labels1=f['labels'], f['data_array']
        images=np.reshape(dnase_data_labels1[1], (batch_size, data_length, 4, 1))
        labels=dnase_data_labels1[0]
    return images, labels

start=time.time()


try:
    options, args =getopt.getopt(sys.argv[1:], 'm:t:n:o:d:', ['model=','test_batch=','network_constructor=','output_dir=','deconv='])
except getopt.GetoptError as err:
    #print str(err)
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


#output_dir=None

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

keep_prob = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)


x_image = tf.placeholder(tf.float32, shape=[None, 1000, 4, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 20])
phase=tf.placeholder(tf.bool)
dropout_1=0.95
dropout_2=0.9
dropout_3=0.85
batch_size=100
data_length=1000 
input_dir=trained_model
nc=il.import_module("network_constructors."+str(network_constructor))   
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
                 data_length=data_length)


sess.run(tf.global_variables_initializer())
saver=model.saver
saver.restore(sess, input_dir)


batch = test_batch(test_batch_file)
test_accuracy1, y_label1, y_prediction1 =sess.run([model.error, y_, model.prediction[1]], feed_dict={x_image: batch[0], y_: batch[1], keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0}) 
#print "test accuracy (true:false=5:5): "+str(test_accuracy1)
#print deconv

if deconv=="transpose":
    
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

    index_of_image=0
    positive_image=[]
    for y in batch[1]:
        #print y[0]
        if np.sum(y)>0:
            positive_image.append(index_of_image)
        index_of_image+=1
    #print len(positive_image)
    for k in range(len(positive_image)):  
    
        images4=np.reshape(batch[0][positive_image[k]], (1, data_length, 4, 1))
        
        #h_conv3_, h_conv25_, h_conv24_, h_conv23_, h_conv22_,h_conv21_, h_conv2_, h_conv1_, b_conv3_=sess.run([h_conv3, h_conv25,h_conv24, h_conv23, h_conv22,h_conv21, h_conv2, h_conv1, b_conv3], 
        
        #                                                                                                      feed_dict={x_image: images4, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0})
        y,y_sgm, variables_dict, neurons_dict, _3=sess.run(model.prediction, feed_dict={x_image: images4, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0, phase: False})
        """
            variable_dict={"W_conv1": W_conv1, 
                           "W_conv2": W_conv2,
                           "W_conv21": W_conv21, 
                           "W_conv22": W_conv22, 
                           "W_fc1": W_fc1,
                           "W_fc4": W_fc4, 
                           "b_fc1": b_fc1, 
                           "b_fc4": b_fc4}
            neurons_dict={"h_conv22":h_conv22,
                          "h_conv21":h_conv21, 
                          "h_conv2":h_conv2,
                          "h_conv11":h_conv11,
                          "h_conv12":h_conv12,
                          "h_fc1_drop": h_fc1_drop,
                          "h_pool3_flat":h_pool3_flat,
                          "h_pool22":h_pool22,
                          "h_pool21":h_pool21,
                          "h_pool2":h_pool2,
                          "h_pool1":h_pool1,
                          "h_pool1_rc":h_pool1_rc}
        """

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
        h_pool21_t4= conv2d_tp(h_conv22_, variables_dict["W_conv22"], h_pool21_shape)
        _, mask21=max_pool_2x1(h_conv21_)
        #h_unpool21_t4=unpooling.unpool(h_pool21_t4, mask21,output_shape=h_conv21_.shape)
        h_unpool21_t4=unpooling.unpool2(h_pool21_t4, mask21)
        
        h_pool2_shape=list(h_pool2_.shape)
        h_pool2_t4= conv2d_tp(h_unpool21_t4, variables_dict["W_conv21"], h_pool2_shape)
        _, mask2=max_pool_2x1(h_conv2_)
        #h_unpool2_t4=unpooling.unpool(h_pool2_t4,mask2,output_shape=h_conv2_.shape)
        h_unpool2_t4=unpooling.unpool2(h_pool2_t4,mask2)
        
        h_pool1_shape=list(h_pool1_.shape)
        h_pool1_t4= conv2d_tp(h_unpool2_t4, variables_dict["W_conv2"], h_pool1_shape)
        _,mask1=max_pool_2x1(h_conv11_)
        #h_unpool1_t4=unpooling.unpool(h_pool1_t4,mask1,output_shape=h_conv11_.shape)
        h_unpool1_t4=unpooling.unpool2(h_pool1_t4,mask1)
        
        h_pool1_rc_t4=conv2d_tp(h_unpool2_t4, tf.reverse(variables_dict["W_conv2"], [0,1]), h_pool1_shape)
        _,mask1rc=max_pool_2x1(h_conv12_)
        #h_unpool1_rc_t4=unpooling.unpool(h_pool1_rc_t4,mask1rc,output_shape=h_conv12_.shape)
        h_unpool1_rc_t4=unpooling.unpool2(h_pool1_rc_t4,mask1rc)
        
        reconstruction_shape=images4.shape
        #print reconstruction_shape
        reconstruction_conv22=conv2d_tp(h_unpool1_t4, variables_dict["W_conv1"], reconstruction_shape)+conv2d_tp(h_unpool1_rc_t4, tf.reverse(variables_dict["W_conv1"], [0,1]), reconstruction_shape)
        
        sess2.run(tf.global_variables_initializer())
        units_conv22 = sess2.run(reconstruction_conv22)
        

        reshaped_conv22=np.reshape(units_conv22, (data_length, 4))
        
        # Compute and plot first dendrogram.
        fig = plt.figure(figsize=(12,8))
        
        # Plot distance matrix.

        
        axmatrix_conv22 = fig.add_axes([0.05,0.05,0.1,0.9])
        im_conv22 = axmatrix_conv22.matshow(reshaped_conv22, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
        axmatrix_conv22.set_xticks([])
        axmatrix_conv22.set_yticks([])
        axcolor = fig.add_axes([0.16,0.05,0.02,0.9])
        pylab.colorbar(im_conv22, cax=axcolor)
        
        reshaped2=np.reshape(images4, (data_length, 4))
        axmatrix3 = fig.add_axes([0.85,0.05,0.1,0.9])
        im3 = axmatrix3.matshow(reshaped2, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
        axmatrix3.set_xticks([])
        axmatrix3.set_yticks([])
        axcolor = fig.add_axes([0.96,0.05,0.02,0.9])
        pylab.colorbar(im3, cax=axcolor)
        
        np.savez_compressed(str(output_dir)+str(trained_model.split('/')[-1])+"_transpose_"+str(k), 
                            conv22=reshaped_conv22, 
                            original=np.reshape(images4,(data_length, 4)))

        fig.savefig(str(output_dir)+str(trained_model.split('/')[-1])+'_reconstruction_'+str(k)+'.png')
        #plt.show()
        sess2.close()
    sess.close()
elif deconv=="train":
    def recon_variable(shape, variable_name):
        initial = tf.truncated_normal(shape, mean=0.50, stddev=0.5)
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
        
    _1, _2, variavl_dict, neurons_dict,_3=sess.run(model.prediction, feed_dict={x_image: batch[0], keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0})
    fc1_param=model.fc1_param
    dimension22=model.dimension22

    sess.close()
    
    sess2 = tf.Session()
        
    x_image_recon = recon_variable([1, 1000, 4, 1], 'x_image_recon')
    
    h_conv11_re=conv2d_1(x_image_recon, variavl_dict["W_conv1"])
    h_conv12_re=conv2d_1(x_image_recon, tf.reverse(variavl_dict["W_conv1"], [0,1]))
    h_conv11_re_ = tf.nn.relu(h_conv11_re)
    h_conv12_re_ = tf.nn.relu(h_conv12_re)
    h_pool1_re = max_pool_2x2(h_conv11_re_)
    h_pool1_rc_re = max_pool_2x2(h_conv12_re_)
    h_conv2_re = tf.add(tf.nn.relu(conv2d_1(h_pool1_re, variavl_dict["W_conv2"])), tf.nn.relu(conv2d_1(h_pool1_rc_re, tf.reverse(variavl_dict["W_conv2"], [0,1]))))
    h_pool2_re = max_pool_2x2(h_conv2_re)
    h_conv21_re = tf.nn.relu(conv2d_1(h_pool2_re, variavl_dict["W_conv21"]))
    h_pool21_re = max_pool_2x2(h_conv21_re)
    h_conv22_re = tf.nn.relu(conv2d_1(h_pool21_re, variavl_dict["W_conv22"]))
    h_pool22_re = max_pool_4x1(h_conv22_re)
    
    h_pool3_flat_re = tf.reshape(h_pool22_re, [-1, 1*fc1_param*dimension22])

    h_fc1_re = tf.nn.relu(tf.add(tf.matmul(h_pool3_flat_re, variavl_dict["W_fc1"]), variavl_dict["b_fc1"]))
    y_conv_re=tf.add(tf.matmul(h_fc1_re,variavl_dict["W_fc4"]), variavl_dict["b_fc4"])
    #print y_conv_re.shape
    #cost =-tf.reshape(y_conv_re,[1])+tf.reduce_sum(tf.square(x_image_recon))/500.0
    #cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][1])-tf.nn.sigmoid(y_conv_re[0][0]),[1])+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    #cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][1])-tf.nn.sigmoid(y_conv_re[0][0]),[1])*tf.log(tf.nn.sigmoid(y_conv_re[0][1]+0.000001)/(tf.nn.sigmoid(y_conv_re[0][0])+0.000001))+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    #cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][1])*tf.nn.sigmoid(y_conv_re[0][0]),[1])+tf.reduce_sum(tf.square(x_image_recon))/2000.0
    cost =-tf.reshape(tf.nn.sigmoid(y_conv_re[0][1])/(tf.nn.sigmoid(y_conv_re[0][0])+0.00001),[1])+tf.reduce_sum(tf.square(x_image_recon))/2000.0
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
            print('step '+str(i)+' cost: '+str(progress)+', y_conv_re: '+str(y_val))
        #if progress<=-0.9999:
            #break
        if i>20000:
            if cost_list[i-500]-cost_list[i]<=0.0000001:
                print(cost_list[i-100]-cost_list[i])
                break
        #print str(a.shape)+'\n'+str(b.shape)+'\n'+str(1*fc1_param*dimension3)
 
        
            #print a
    final_recon=x_image_recon.eval(session=sess2)
    final_recon_res=np.reshape(final_recon, (data_length, 4))
    inputdir=input_dir.split('/')[-1]
    np.savez_compressed(str(trained_model)+str(start_at), recon=final_recon_res)
    
    fig = plt.figure(figsize=(8,8))
        # Plot distance matrix.
    axmatrix = fig.add_axes([0.1,0.05,0.6,0.9])
    im = axmatrix.matshow(final_recon_res, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    axcolor = fig.add_axes([0.8,0.05,0.02,0.9])
    pylab.colorbar(im, cax=axcolor)
    fig.savefig(str(trained_model)+'_max_act_'+str(start_at)+'.png')
    import deepgmap.post_train_tools.sequence_visualizer2 as sq
    sq.seuquence_visualizer2(final_recon_res, str(trained_model)+'_max_act_seq_'+str(start_at)+'.pdf')
    plt.show()
    sess2.close()
else:
    print("don't understand the "+str(deconv)+" option")
                                               