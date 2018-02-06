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
import network_constructor as nc



def next_batch(loop):

    with gzip.open('/media/koh/HD-PCFU3/mouse/various_dnase_data/50_0_shuf/dnase+random+labels_multi_filter_rand_50_0_shuf_'+str(loop)+'.cpickle.gz', 'r') as f1:
        dnase_data_labels=cPickle.load(f1)
        images=np.reshape(dnase_data_labels[1], (batch_size, data_length, 4, 1))
        labels=dnase_data_labels[0]
    return images, labels

def Three_label_ROCspace_calculator(a, b):
    True_positive=(2, 0)
    True_negative=(0, 2)
    False_postive=(-1, 1)
    False_negative=(1, -1)
    total_false=(1,1)
    ROC_counter=np.array([0,0,0,0], np.int32)
    for i in range(len(a)):
        b1=[0,0]
        index=np.argmax(b[i])
        b1[index]+=1  
        c=a[i]+b1
        d=a[i]-b1
        if (c==True_positive).all():
            ROC_counter+=[1,0,0,0]
        elif (c==True_negative).all():
            ROC_counter+=[0,0,1,0]
        elif (d==False_postive).all():
            ROC_counter+=[0,1,0,0]
        elif (d==False_negative).all():
            ROC_counter+=[0,0,0,1]    
    FPR=float(ROC_counter[1])/(float(ROC_counter[2])+float(ROC_counter[1])+0.001)
    FNR=float(ROC_counter[3])/(float(ROC_counter[0])+float(ROC_counter[3])+0.001)  
    return FPR, FNR 
def test_batch():
        with gzip.open('/media/koh/HD-PCFU3/mouse/various_dnase_data/cne_dnase_50_15_shuf2/dnase+random+labels_cne_dnase_filter_rand_shuf_2_50_15_3501.cpickle.gz') as f2, gzip.open('/media/koh/HD-PCFU3/mouse/dnase_genome/100/dnase+random+labels1501.cpickle.gz') as f3, gzip.open('/media/koh/HD-PCFU3/mouse/dnase/100/dnase+random+labels1501.cpickle.gz') as f4:
            dnase_data_labels1=cPickle.load(f2)
            dnase_data_labels2=cPickle.load(f3)
            dnase_data_labels3=cPickle.load(f4)
            images1=np.reshape(dnase_data_labels1[1], (batch_size, data_length, 4, 1))
            images2=np.reshape(dnase_data_labels2[1], (batch_size, data_length, 4, 1))
            images3=np.reshape(dnase_data_labels3[1], (batch_size, data_length, 4, 1))
            labels1=dnase_data_labels1[0]
            labels2=dnase_data_labels2[0]
            labels3=dnase_data_labels3[0]
        return images1, labels1, images2, labels2, images3, labels3

start=time.time()

sess = tf.Session()
x_image = tf.placeholder(tf.float32, shape=[None, 1000, 4, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 1])
keep_prob = tf.placeholder(tf.float32)
keep_prob2 = tf.placeholder(tf.float32)
keep_prob3 = tf.placeholder(tf.float32)
dropout_1=0.95
dropout_2=0.9
dropout_3=0.85
batch_size=100
data_length=1000 
input_dir=sys.argv[1].strip('.log')    
train_speed=0.00005
a=time.asctime()
b=a.replace(':', '')
start_at=b.replace(' ', '_')

model = nc.Model(image=x_image, label=y_, output_dir=None, start_at=start_at, keep_prob=keep_prob, keep_prob2=keep_prob2, keep_prob3=keep_prob3, data_length=data_length)
sess.run(tf.initialize_all_variables())
saver=model.saver
saver.restore(sess, input_dir)


batch = test_batch()
test_accuracy1, y_label1, y_prediction1 =sess.run([model.error, y_, model.prediction[1]], feed_dict={x_image: batch[0], y_: batch[1], keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0}) 
print "test accuracy (true:false=5:5): "+str(test_accuracy1)
transpose=False
if transpose==True:
    
    def conv2d_tp(x, W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 1, 1, 1], padding='VALID')
    def conv2d_tp2(x, W, output_shape):
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 1, 1], padding='VALID') 
    
    index_of_image=0
    positive_image=[]
    for y in batch[1]:
        if (y==[1,0]).all():
            positive_image.append(index_of_image)
        index_of_image+=1
    
    for k in range(len(positive_image)):  
    
        images4=np.reshape(batch[0][positive_image[k]], (1, data_length, 4, 1))
        
        #h_conv3_, h_conv25_, h_conv24_, h_conv23_, h_conv22_,h_conv21_, h_conv2_, h_conv1_, b_conv3_=sess.run([h_conv3, h_conv25,h_conv24, h_conv23, h_conv22,h_conv21, h_conv2, h_conv1, b_conv3], 
        
        #                                                                                                      feed_dict={x_image: images4, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0})
        _, variavl_dict, neurons_dict=sess.run(model.prediction, feed_dict={x_image: images4, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0})
        
        h_conv3_, h_conv25_, h_conv24_, h_conv23_, h_conv22_,h_conv21_, h_conv2_, h_conv1_, b_conv3_=(neurons_dict["h_conv3"], 
                                                                                                      neurons_dict["h_conv25"], 
                                                                                                      neurons_dict["h_conv24"], 
                                                                                                      neurons_dict["h_conv23"],
                                                                                                      neurons_dict["h_conv22"],
                                                                                                      neurons_dict["h_conv21"],
                                                                                                      neurons_dict["h_conv2"],
                                                                                                      neurons_dict["h_conv1"],
                                                                                                      variavl_dict["b_conv3"])
        
        max_index_h_conv3_=np.argmax(h_conv3_)
        print max_index_h_conv3_
        shape_of_h_conv3_=list(h_conv3_.shape)
        shape_of_b_conv3_=list(b_conv3_.shape)
        
        #true_index1=max_index_h_conv3_/shape_of_h_conv3_[3]
        #true_index2=max_index_h_conv3_%shape_of_h_conv3_[3]
        #max_values=np.amax(h_conv3_)
        #h_conv3_sparse=np.zeros(shape_of_h_conv3_, np.float32)
        #h_conv3_sparse[0][true_index1][0][true_index2]+=max_values
        #print np.argmax(h_conv3_sparse)
        h_conv25_shape=list(h_conv25_.shape)
        h_conv25_t= tf.nn.relu(conv2d_tp2(h_conv3_-variavl_dict["b_conv3"], variavl_dict["W_conv3"], h_conv25_shape))
        h_conv24_shape=list(h_conv24_.shape)
        h_conv24_t= tf.nn.relu(conv2d_tp2(h_conv25_t-variavl_dict["b_conv25"], variavl_dict["W_conv25"], h_conv24_shape))
        h_conv23_shape=list(h_conv23_.shape)
        h_conv23_t= tf.nn.relu(conv2d_tp2(h_conv24_t-variavl_dict["b_conv24"], variavl_dict["W_conv24"], h_conv23_shape))
        h_conv22_shape=list(h_conv22_.shape)
        h_conv22_t= tf.nn.relu(conv2d_tp2(h_conv23_t-variavl_dict["b_conv23"], variavl_dict["W_conv23"], h_conv22_shape))
        h_conv21_shape=list(h_conv21_.shape)
        h_conv21_t= tf.nn.relu(conv2d_tp2(h_conv22_t-variavl_dict["b_conv22"], variavl_dict["W_conv22"], h_conv21_shape))
        h_conv2_shape=list(h_conv2_.shape)
        h_conv2_t= tf.nn.relu(conv2d_tp2(h_conv21_t-variavl_dict["b_conv21"], variavl_dict["W_conv21"], h_conv2_shape))
        h_conv1_shape=list(h_conv1_.shape)
        h_conv1_t= tf.nn.relu(conv2d_tp2(h_conv2_t-variavl_dict["b_conv2"], variavl_dict["W_conv2"], h_conv1_shape))
        reconstruction_shape=images4.shape
        #reconstruction_conv3=tf.nn.relu(conv2d_tp(h_conv1_t-variavl_dict["b_conv1"], variavl_dict["W_conv1"], reconstruction_shape))
        reconstruction_conv3=tf.nn.relu(conv2d_tp(h_conv1_t-variavl_dict["b_conv1"], variavl_dict["W_conv1"], reconstruction_shape)
                                        +conv2d_tp(h_conv1_t-variavl_dict["b_conv1"], tf.reverse(variavl_dict["W_conv1"], [True, True,False,False]), reconstruction_shape))
        
        h_conv24_shape=list(h_conv24_.shape)
        h_conv24_t1= tf.nn.relu(conv2d_tp2(h_conv25_-variavl_dict["b_conv25"], variavl_dict["W_conv25"], h_conv24_shape))
        h_conv23_shape=list(h_conv23_.shape)
        h_conv23_t1= tf.nn.relu(conv2d_tp2(h_conv24_t1-variavl_dict["b_conv24"], variavl_dict["W_conv24"], h_conv23_shape))
        h_conv22_shape=list(h_conv22_.shape)
        h_conv22_t1= tf.nn.relu(conv2d_tp2(h_conv23_t1-variavl_dict["b_conv23"], variavl_dict["W_conv23"], h_conv22_shape))
        h_conv21_shape=list(h_conv21_.shape)
        h_conv21_t1= tf.nn.relu(conv2d_tp2(h_conv22_t1-variavl_dict["b_conv22"], variavl_dict["W_conv22"], h_conv21_shape))
        h_conv2_shape=list(h_conv2_.shape)
        h_conv2_t1= tf.nn.relu(conv2d_tp2(h_conv21_t1-variavl_dict["b_conv21"], variavl_dict["W_conv21"], h_conv2_shape))
        h_conv1_shape=list(h_conv1_.shape)
        h_conv1_t1= tf.nn.relu(conv2d_tp2(h_conv2_t1-variavl_dict["b_conv2"], variavl_dict["W_conv2"], h_conv1_shape))
        reconstruction_shape=images4.shape
        reconstruction_conv25=tf.nn.relu(conv2d_tp(h_conv1_t1-variavl_dict["b_conv1"], variavl_dict["W_conv1"], reconstruction_shape)
                                         +conv2d_tp(h_conv1_t1-variavl_dict["b_conv1"], tf.reverse(variavl_dict["W_conv1"], [True, True,False,False]), reconstruction_shape))
        
        h_conv23_shape=list(h_conv23_.shape)
        h_conv23_t2= tf.nn.relu(conv2d_tp2(h_conv24_-variavl_dict["b_conv24"], variavl_dict["W_conv24"], h_conv23_shape))
        h_conv22_shape=list(h_conv22_.shape)
        h_conv22_t2= tf.nn.relu(conv2d_tp2(h_conv23_t2-variavl_dict["b_conv23"], variavl_dict["W_conv23"], h_conv22_shape))
        h_conv21_shape=list(h_conv21_.shape)
        h_conv21_t2= tf.nn.relu(conv2d_tp2(h_conv22_t2-variavl_dict["b_conv22"], variavl_dict["W_conv22"], h_conv21_shape))
        h_conv2_shape=list(h_conv2_.shape)
        h_conv2_t2= tf.nn.relu(conv2d_tp2(h_conv21_t2-variavl_dict["b_conv21"], variavl_dict["W_conv21"], h_conv2_shape))
        h_conv1_shape=list(h_conv1_.shape)
        h_conv1_t2= tf.nn.relu(conv2d_tp2(h_conv2_t2-variavl_dict["b_conv2"], variavl_dict["W_conv2"], h_conv1_shape))
        reconstruction_shape=images4.shape
        reconstruction_conv24=tf.nn.relu(conv2d_tp(h_conv1_t2-variavl_dict["b_conv1"], variavl_dict["W_conv1"], reconstruction_shape)
                                         +conv2d_tp(h_conv1_t2-variavl_dict["b_conv1"], tf.reverse(variavl_dict["W_conv1"], [True, True,False,False]), reconstruction_shape))
        
        h_conv22_shape=list(h_conv22_.shape)
        h_conv22_t3= tf.nn.relu(conv2d_tp2(h_conv23_-variavl_dict["b_conv23"], variavl_dict["W_conv23"], h_conv22_shape))
        h_conv21_shape=list(h_conv21_.shape)
        h_conv21_t3= tf.nn.relu(conv2d_tp2(h_conv22_t3-variavl_dict["b_conv22"], variavl_dict["W_conv22"], h_conv21_shape))
        h_conv2_shape=list(h_conv2_.shape)
        h_conv2_t3= tf.nn.relu(conv2d_tp2(h_conv21_t3-variavl_dict["b_conv21"], variavl_dict["W_conv21"], h_conv2_shape))
        h_conv1_shape=list(h_conv1_.shape)
        h_conv1_t3= tf.nn.relu(conv2d_tp2(h_conv2_t3-variavl_dict["b_conv2"], variavl_dict["W_conv2"], h_conv1_shape))
        reconstruction_shape=images4.shape
        reconstruction_conv23=tf.nn.relu(conv2d_tp(h_conv1_t3-variavl_dict["b_conv1"], variavl_dict["W_conv1"], reconstruction_shape)
                                         +conv2d_tp(h_conv1_t3-variavl_dict["b_conv1"], tf.reverse(variavl_dict["W_conv1"], [True, True,False,False]), reconstruction_shape))
        
        h_conv21_shape=list(h_conv21_.shape)
        h_conv21_t4= tf.nn.relu(conv2d_tp2(h_conv22_-variavl_dict["b_conv22"], variavl_dict["W_conv22"], h_conv21_shape))
        h_conv2_shape=list(h_conv2_.shape)
        h_conv2_t4= tf.nn.relu(conv2d_tp2(h_conv21_t4-variavl_dict["b_conv21"], variavl_dict["W_conv21"], h_conv2_shape))
        h_conv1_shape=list(h_conv1_.shape)
        h_conv1_t4= tf.nn.relu(conv2d_tp2(h_conv2_t4-variavl_dict["b_conv2"], variavl_dict["W_conv2"], h_conv1_shape))
        reconstruction_shape=images4.shape
        reconstruction_conv22=tf.nn.relu(conv2d_tp(h_conv1_t4-variavl_dict["b_conv1"], variavl_dict["W_conv1"], reconstruction_shape)
                                         +conv2d_tp(h_conv1_t4-variavl_dict["b_conv1"], tf.reverse(variavl_dict["W_conv1"], [True, True,False,False]), reconstruction_shape))
        
        h_conv2_shape=list(h_conv2_.shape)
        h_conv2_t5= tf.nn.relu(conv2d_tp2(h_conv21_-variavl_dict["b_conv21"], variavl_dict["W_conv21"], h_conv2_shape))
        h_conv1_shape=list(h_conv1_.shape)
        h_conv1_t5= tf.nn.relu(conv2d_tp2(h_conv2_t5-variavl_dict["b_conv2"], variavl_dict["W_conv2"], h_conv1_shape))
        reconstruction_shape=images4.shape
        reconstruction_conv21=tf.nn.relu(conv2d_tp(h_conv1_t5-variavl_dict["b_conv1"], variavl_dict["W_conv1"], reconstruction_shape)
                                         +conv2d_tp(h_conv1_t5-variavl_dict["b_conv1"], tf.reverse(variavl_dict["W_conv1"], [True, True,False,False]), reconstruction_shape))
        
        h_conv1_shape=list(h_conv1_.shape)
        h_conv1_t6= tf.nn.relu(conv2d_tp2(h_conv2_-variavl_dict["b_conv2"], variavl_dict["W_conv2"], h_conv1_shape))
        reconstruction_shape=images4.shape
        reconstruction_conv2=tf.nn.relu(conv2d_tp(h_conv1_t6, variavl_dict["W_conv1"], reconstruction_shape)
                                        +conv2d_tp(h_conv1_t6-variavl_dict["b_conv1"], tf.reverse(variavl_dict["W_conv1"], [True, True,False,False]), reconstruction_shape))
        
        reconstruction_conv1=tf.nn.relu(conv2d_tp(h_conv1_-variavl_dict["b_conv1"], variavl_dict["W_conv1"], reconstruction_shape)
                                        +conv2d_tp(h_conv1_-variavl_dict["b_conv1"], tf.reverse(variavl_dict["W_conv1"], [True, True,False,False]), reconstruction_shape))

        units = reconstruction_conv3.eval(session=sess)
        units_conv25 = reconstruction_conv25.eval(session=sess)
        units_conv24 = reconstruction_conv24.eval(session=sess)
        units_conv23 = reconstruction_conv23.eval(session=sess)
        units_conv22 = reconstruction_conv22.eval(session=sess)
        units_conv21 = reconstruction_conv21.eval(session=sess)
        units_conv2 = reconstruction_conv2.eval(session=sess)
        units_conv1 = reconstruction_conv1.eval(session=sess)
        
        reshaped=np.reshape(units, (data_length, 4))
        reshaped_conv1=np.reshape(units_conv1, (data_length, 4))
        reshaped_conv2=np.reshape(units_conv2, (data_length, 4))
        reshaped_conv21=np.reshape(units_conv21, (data_length, 4))    
        reshaped_conv22=np.reshape(units_conv22, (data_length, 4))
        reshaped_conv23=np.reshape(units_conv23, (data_length, 4))
        reshaped_conv24=np.reshape(units_conv24, (data_length, 4))
        reshaped_conv25=np.reshape(units_conv25, (data_length, 4))         
        # Compute and plot first dendrogram.
        fig = plt.figure(figsize=(12,8))
        
        # Plot distance matrix.
        axmatrix = fig.add_axes([0.05,0.05,0.1,0.9])
        im = axmatrix.matshow(reshaped, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])
        
        axmatrix_conv25 = fig.add_axes([0.15,0.05,0.1,0.9])
        im_conv25 = axmatrix_conv25.matshow(reshaped_conv25, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
        axmatrix_conv25.set_xticks([])
        axmatrix_conv25.set_yticks([])
        
        axmatrix_conv24 = fig.add_axes([0.25,0.05,0.1,0.9])
        im_conv24 = axmatrix_conv24.matshow(reshaped_conv24, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
        axmatrix_conv24.set_xticks([])
        axmatrix_conv24.set_yticks([])
        
        axmatrix_conv23 = fig.add_axes([0.35,0.05,0.1,0.9])
        im_conv23 = axmatrix_conv23.matshow(reshaped_conv23, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
        axmatrix_conv23.set_xticks([])
        axmatrix_conv23.set_yticks([])
        
        axmatrix_conv22 = fig.add_axes([0.45,0.05,0.1,0.9])
        im_conv22 = axmatrix_conv22.matshow(reshaped_conv22, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
        axmatrix_conv22.set_xticks([])
        axmatrix_conv22.set_yticks([])
        
        axmatrix_conv21 = fig.add_axes([0.55,0.05,0.1,0.9])
        im_conv21 = axmatrix_conv21.matshow(reshaped_conv21, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
        axmatrix_conv21.set_xticks([])
        axmatrix_conv21.set_yticks([])
        
        axmatrix_conv2 = fig.add_axes([0.65,0.05,0.1,0.9])
        im_conv2 = axmatrix_conv2.matshow(reshaped_conv2, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
        axmatrix_conv2.set_xticks([])
        axmatrix_conv2.set_yticks([])
        
        axmatrix_conv1 = fig.add_axes([0.75,0.05,0.1,0.9])
        im_conv1 = axmatrix_conv1.matshow(reshaped_conv1, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
        axmatrix_conv1.set_xticks([])
        axmatrix_conv1.set_yticks([])
        
        reshaped2=np.reshape(images4, (data_length, 4))
        axmatrix3 = fig.add_axes([0.85,0.05,0.1,0.9])
        im3 = axmatrix3.matshow(reshaped2, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
        axmatrix3.set_xticks([])
        axmatrix3.set_yticks([])
        
        # Plot colorbar.
        axcolor = fig.add_axes([0.96,0.05,0.02,0.9])
        pylab.colorbar(im, cax=axcolor)
        with gzip.open('/media/koh/HD-PCFU3/mouse/reconstruction_and_original_'+str(k)+'_'+str(start_at)+'.cpickle.gz', 'w') as outfile1:
            #saving all posterior variables into outfile2
            cPickle.dump([reshaped, reshaped_conv24, reshaped2], outfile1, protocol=cPickle.HIGHEST_PROTOCOL)
            
        fig.savefig('/media/koh/HD-PCFU3/mouse/reconstruction_'+str(k)+'_'+str(start_at)+'.png')
        plt.show()
else:
    def recon_variable(shape, variable_name):
        initial = tf.truncated_normal(shape, mean=0.5, stddev=0.01)
        return tf.Variable(initial, name=variable_name)
    _, variavl_dict, neurons_dict=sess.run(model.prediction, feed_dict={x_image: batch[0], keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0})
    fc1_param=model.fc1_param
    dimension3=model.dimension3
    def conv2d_1(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 2, 1, 1], padding='VALID')      
    sess.close()
    sess2 = tf.Session()
    x_image_recon = recon_variable([1, 1000, 4, 1], 'x_image_recon')
    h_conv1_re = tf.nn.relu(conv2d_1(x_image_recon, variavl_dict["W_conv1"]) + variavl_dict["b_conv1"])
    h_conv2_re = tf.nn.relu(conv2d(h_conv1_re, variavl_dict["W_conv2"]) + variavl_dict["b_conv2"])
    h_conv21_re = tf.nn.relu(conv2d(h_conv2_re, variavl_dict["W_conv21"]) + variavl_dict["b_conv21"])
    h_conv22_re = tf.nn.relu(conv2d(h_conv21_re, variavl_dict["W_conv22"]) + variavl_dict["b_conv22"])
    h_conv23_re = tf.nn.relu(conv2d(h_conv22_re, variavl_dict["W_conv23"]) + variavl_dict["b_conv23"])
    h_conv24_re = tf.nn.relu(conv2d(h_conv23_re, variavl_dict["W_conv24"]) + variavl_dict["b_conv24"])
    h_conv25_re = tf.nn.relu(conv2d(h_conv24_re, variavl_dict["W_conv25"]) + variavl_dict["b_conv25"])
    h_conv3_re = tf.nn.relu(conv2d(h_conv25_re, variavl_dict["W_conv3"]) + variavl_dict["b_conv3"])
    h_pool3_flat_re = tf.reshape(h_conv3_re, [-1, 1*fc1_param*dimension3])
    h_fc1_re = tf.nn.relu(tf.matmul(h_pool3_flat_re, variavl_dict["W_fc1"]) + variavl_dict["b_fc1"])
    h_fc3_re = tf.nn.relu(tf.matmul(h_fc1_re, variavl_dict["W_fc3"]) + variavl_dict["b_fc3"]) 
    y_conv_re=tf.nn.softmax(tf.matmul(h_fc3_re,variavl_dict["W_fc4"]) + variavl_dict["b_fc4"])
    sess2.run(tf.initialize_all_variables())
    cost =-( tf.reduce_sum(tf.multiply(y_conv_re, [1,0]))-tf.reduce_sum(tf.square(x_image_recon))/100.0)
    train_step2 = tf.train.GradientDescentOptimizer(0.2).minimize(cost)
    for i in range(10000):
        _, progress, y_val, a, b  =sess2.run([train_step2, cost, y_conv_re, h_pool3_flat_re, h_conv3_re])
        #print str(a.shape)+'\n'+str(b.shape)+'\n'+str(1*fc1_param*dimension3)
        if i%100==0:
            print 'step '+str(i)+' cost: '+str(progress)+', y_conv_re: '+str(y_val)
    final_recon=x_image_recon.eval(session=sess2)
    final_recon_res=np.reshape(final_recon, (data_length, 4))
    fig = plt.figure(figsize=(8,8))
        # Plot distance matrix.
    axmatrix = fig.add_axes([0.1,0.05,0.6,0.9])
    im = axmatrix.matshow(final_recon_res, aspect='auto', origin='lower', cmap=plt.get_cmap('YlGnBu'))
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    axcolor = fig.add_axes([0.8,0.05,0.02,0.9])
    pylab.colorbar(im, cax=axcolor)
    fig.savefig('/media/koh/HD-PCFU3/mouse/reconstruction_max_act_'+str(start_at)+'.png')
    plt.show()
    sess2.close()
                                               