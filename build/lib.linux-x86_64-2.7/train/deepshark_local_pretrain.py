
import sys
import gzip
import cPickle
import tensorflow as tf
import numpy as np
import time
import math
start=time.time()
#dimension1_2=16
sess = tf.Session()
x_image = tf.placeholder(tf.float32, shape=[None, 1000, 4, 1])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

# parameter lists
initial_variation=0.05 #standard deviation of initial variables in the convolution filters
batch_size=100 #mini batch size
dimension1=512 #the number of the convolution filters in the 1st layer
dimension2=64 #the number of the convolution filters in the 2nd layer
dimension3=128 #the number of the convolution filters in the 3rd layer
dimension4=1024 #the number of the neurons in each layer of the fully-connected neural network
data_length=1000 
conv1_filter=21
#conv1_filter2=49
conv2_filter=11
conv3_filter=7
train_speed=0.0001
a=time.asctime()
b=a.replace(':', '')
start_at=b.replace(' ', '_')


"""file_list=sys.argv
training_data=file_list[1]
test_data1=file_list[2]
test_data2=file_list[3]
output_dir=file_list[4]"""

flog=open('/media/koh/HD-PCFU3/mouse/'+start_at+'.log', 'w')
flog.write("the filer number of conv1:"+ str(dimension1)+"\n"
          +"the filer size of conv1:"+ str(conv1_filter)+"\n"
          +"the filer number of conv2:"+ str(dimension2)+"\n"
          +"the filer size of conv2:"+ str(conv2_filter)+"\n"
          +"the filer number of conv3:"+ str(dimension3)+"\n"
          +"the filer size of conv3:"+ str(conv3_filter)+"\n"
          +"the number of neurons in the fully-connected layer:"+ str(dimension4)+"\n"
          +"the standard deviation of initial varialbles:"+ str(initial_variation)+"\n"
          +"train speed:"+ str(train_speed)+"\n"
          +"data length:" + str(data_length)+"\n"
          +"batch size:"+str(batch_size)+"\n")
flog.close()

def weight_variable(shape, variable_name):
    initial = tf.truncated_normal(shape, mean=0, stddev=initial_variation)
    return tf.Variable(initial, name=variable_name)
  
def bias_variable(shape, variable_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=variable_name)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')
def max_pool_4x1(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 1, 1], strides=[1, 4, 1, 1], padding='SAME')
def max_pool_8x1(x):
    return tf.nn.max_pool(x, ksize=[1, 17, 1, 1], strides=[1, 17, 1, 1], padding='SAME')
def next_batch(loop):

    with gzip.open('/media/koh/HD-PCFU3/mouse/various_dnase_data/art_rand/100/dnase+random+labels_multi_filter_shuf'+str(loop)+'.cpickle.gz', 'r') as f1:
        dnase_data_labels=cPickle.load(f1)
        images=np.reshape(dnase_data_labels[1], (batch_size, data_length, 4, 1))
        labels=dnase_data_labels[0]
    return images, labels
def next_batch_for_pretrain(loop):
    import glob
    f = glob.glob("/media/koh/HD-PCFU3/mouse/pretrain/genome_chr*_standard.cpickle.gz")
    f.sort()
    with gzip.open(f[loop], 'r') as f1:
        data_labels=cPickle.load(f1)
        data_shape=data_labels[1].shape
        images=np.reshape(data_labels[1], (data_shape[0], data_shape[1], 4, 1))
        labels=data_labels[0]
        images_list=np.array_split(images, 100)
        labels_list=np.array_split(labels, 100)
    return images_list, labels_list

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
        
    



fc1_param=int(math.ceil((math.ceil((math.ceil((data_length-conv1_filter+1)/4.0)-conv2_filter+1)/2.0)-conv3_filter+1)/2.0))

    
W_conv1 = weight_variable([conv1_filter, 4, 1, dimension1], 'W_conv1')
b_conv1 = bias_variable([dimension1], 'b_conv1')
norm1=tf.nn.batch_normalization((conv2d(x_image, W_conv1) + b_conv1), mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001)
h_conv1 = tf.nn.relu(norm1)
h_pool1 = max_pool_4x1(h_conv1)

#fc1_param_2=int((math.ceil((data_length-conv1_filter2+1)/8.0)))
#W_conv1_2 = weight_variable([conv1_filter2, 4, 1, dimension1_2])
#b_conv1_2 = bias_variable([dimension1_2])
#norm1_2=tf.nn.batch_normalization((conv2d(x_image, W_conv1) + b_conv1), mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001)
#h_conv1_2 = tf.nn.relu(norm1_2)
#h_pool1_2 = max_pool_4x1(h_conv1_2)
#W_fc1_2 = weight_variable([1 * fc1_param_2 * dimension1_2, dimension4])
#b_fc1_2 = bias_variable([dimension4])

W_conv2 = weight_variable([conv2_filter, 1, dimension1, dimension2], 'W_conv2')
b_conv2 = bias_variable([dimension2], 'b_conv2')
norm2=tf.nn.batch_normalization((conv2d(h_pool1, W_conv2) + b_conv2), mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001)
h_conv2 = tf.nn.relu(norm2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([conv3_filter, 1, dimension2, dimension3], 'W_conv3')
b_conv3 = bias_variable([dimension3], 'b_conv3')
norm3=tf.nn.batch_normalization((conv2d(h_pool2, W_conv3) + b_conv3), mean=0.0, variance=1, offset=0, scale=1, variance_epsilon=0.001)
h_conv3 = tf.nn.relu(norm3)
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = weight_variable([1 * fc1_param * dimension3, dimension4], 'W_fc1')
b_fc1 = bias_variable([dimension4], 'b_fc1')
h_pool3_flat = tf.reshape(h_pool3, [-1, 1*fc1_param*dimension3])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([dimension4, dimension4], 'W_fc2')
b_fc2 = bias_variable([dimension4], 'b_fc2')
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([dimension4, dimension4], 'W_fc3')
b_fc3 = bias_variable([dimension4], 'b_fc3')
h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

W_fc4 = weight_variable([dimension4, 2], 'W_fc4')
b_fc4 = bias_variable([2], 'b_fc4')

y_conv=tf.nn.softmax(tf.matmul(h_fc3, W_fc4) + b_fc4)

#cost = tf.reduce_mean(tf.reduce_sum(tf.square(y_-y_conv), reduction_indices=[1]))
cost =tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv,1e-10,1.0))+(1-y_)*tf.log(tf.clip_by_value(1-y_conv,1e-10,1.0)), reduction_indices=[1]))
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
#cost = tf.reduce_mean(tf.reduce_sum(tf.square(y_conv*tf.log(tf.clip_by_value(2*y_conv,1e-10,1.0)/(tf.clip_by_value(y_conv,1e-10,1.0)+tf.clip_by_value(y_,1e-10,1.0)))+y_*tf.log(2*tf.clip_by_value(y_,1e-10,1.0)/(tf.clip_by_value(y_conv,1e-10,1.0)+tf.clip_by_value(y_,1e-10,1.0)))), reduction_indices=[1]))
#cost = tf.reduce_sum(tf.square(y_conv*tf.log(tf.clip_by_value(2*y_conv,1e-10,1.0)/(tf.clip_by_value(y_conv,1e-10,1.0)+tf.clip_by_value(y_,1e-10,1.0)))+y_*tf.log(2*tf.clip_by_value(y_,1e-10,1.0)/(tf.clip_by_value(y_conv,1e-10,1.0)+tf.clip_by_value(y_,1e-10,1.0)))))
train_step = tf.train.AdamOptimizer(train_speed).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))-0.5)/(1-0.5)
sess.run(tf.initialize_all_variables())
saver=tf.train.Saver(var_list={"W_conv1": W_conv1, "W_conv2": W_conv2, "W_conv3": W_conv3, "b_conv1": b_conv1, "b_conv2": b_conv2, "b_conv3": b_conv3, "W_fc1": W_fc1, "W_fc2": W_fc2, "W_fc3": W_fc3, "W_fc4": W_fc4, "b_fc1": b_fc1, "b_fc2": b_fc2,"b_fc3": b_fc3,"b_fc4": b_fc4})

import matplotlib.pyplot as plt
loop_num=1
train_accuracy_record=np.zeros((loop_num), np.float32)
loss_val_record=np.zeros((loop_num), np.float32)
total_learing=np.zeros((loop_num), np.int32)

for i in range(loop_num):
    batch = next_batch_for_pretrain(i)
    for j in range(len(batch[0])):
        _, loss_val, train_accuracy, current_y=sess.run([train_step, cost, accuracy, y_conv], feed_dict={x_image: batch[0][j], y_: batch[1][j], keep_prob: 0.7})
        train_accuracy_record[i]+=train_accuracy
        loss_val_record[i]+=loss_val
        total_learing[i]+=i*batch_size/1000.0

    
        print "step "+str(i)+", cost: "+str(loss_val)+", train_accuracy: "+str(train_accuracy)
    if i==loop_num-1:
        saver.save(sess, '/media/koh/HD-PCFU3/mouse/pretrained_'+str(start_at)+'.ckpt', global_step=i)
        tf_variables=sess.run([W_conv1, W_conv2, W_conv3, b_conv1, b_conv2, b_conv3, W_fc1, W_fc2, W_fc3, W_fc4, b_fc1, b_fc2, b_fc3, b_fc4])
        with open('/media/koh/HD-PCFU3/mouse/filter1_'+str(i)+'.txt', 'w') as outfile1, gzip.open('/media/koh/HD-PCFU3/mouse/variables_'+str(i)+'_'+str(start_at)+'.cpickle.gz', 'w') as outfile2:
                
                #saving all posterior variables into outfile2
            cPickle.dump(tf_variables, outfile2, protocol=cPickle.HIGHEST_PROTOCOL)
                
            m=0
            k=0
            l=0
                # Writing the convolutional filters of the 1st layer into outfile1 
            for m in range(dimension1):
                outfile1.write('# filter1_'+str(j)+'\n')
                for k in range(4):
                    for l in range(conv1_filter):
                        outfile1.write(str((tf_variables[0])[l][k][0][m])+' ')
                    outfile1.write('\n')
                    
                            
                
"""    def test_batch():
        with gzip.open('/media/koh/HD-PCFU3/mouse/various_dnase_data/art_rand/100/dnase+random+labels_multi_filter_shuf3501.cpickle.gz') as f2, gzip.open('/media/koh/HD-PCFU3/mouse/dnase_genome/100/dnase+random+labels1501.cpickle.gz') as f3, gzip.open('/media/koh/HD-PCFU3/mouse/dnase/100/dnase+random+labels1501.cpickle.gz') as f4:
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
    batch = test_batch()
    test_accuracy1, y_label1, y_prediction1 =sess.run([accuracy, y_, y_conv], feed_dict={x_image: batch[0], y_: batch[1], keep_prob: 1.0})
    test_accuracy2, y_label2, y_prediction2 =sess.run([accuracy, y_, y_conv], feed_dict={x_image: batch[2], y_: batch[3], keep_prob: 1.0})
    test_accuracy3, y_label3, y_prediction3 =sess.run([accuracy, y_, y_conv], feed_dict={x_image: batch[4], y_: batch[5], keep_prob: 1.0})
    FPR1, FNR1 =Three_label_ROCspace_calculator(y_label1, y_prediction1)
    FPR2, FNR2 =Three_label_ROCspace_calculator(y_label2, y_prediction2)
    FPR3, FNR3 =Three_label_ROCspace_calculator(y_label3, y_prediction3)
    to_print="test accuracy (true:false=5:5): "+str(test_accuracy1)+" with FPR, FNR = "+str(FPR1)+", "+str(FPR1)+"\ntest accuracy (random_genome): "+ str(test_accuracy2)+" with FPR, FNR = "+str(FPR2)+", "+str(FNR2)+"\ntest accuracy (artificial random seq): "+ str(test_accuracy3)+" with FPR, FNR = "+str(FPR3)+", "+str(FNR3)+"\n Total time "+ str(time.time()-start)
    
    sess.close()
    print(to_print)
    flog=open('/media/koh/HD-PCFU3/mouse/'+start_at+'.log', 'a')
    flog.write(to_print+'\n')
    flog.close()
    
    fit=np.polyfit(total_learing, train_accuracy_record, 1)
    fit_fn=np.poly1d(fit)
    
    plt.figure(1)
    plt.subplot(211)
    plt.title('Train accuracy')
    #plt.plot(total_learing, train_accuracy_record, 'c.', total_learing, fit_fn(total_learing), 'm-')
    plt.plot(total_learing, train_accuracy_record, 'c.')
    
    plt.figure(1)
    plt.subplot(212)
    plt.title('Cost')
    plt.plot(total_learing,loss_val_record, '-')
    plt.savefig('/media/koh/HD-PCFU3/mouse/plot'+str(start_at)+'.pdf', format='pdf')
    plt.show()

elif sys.argv[1]=='test':
    saver=tf.train.Saver()
    saver.restore(sess, '/media/koh/HD-PCFU3/mouse/'+str(start_at))"""
    
