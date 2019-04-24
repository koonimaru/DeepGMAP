import sys
import gzip
import cPickle
import tensorflow as tf
import numpy as np
import time
import math
import os
from natsort import natsorted, ns
from deepgmap.network_constructors import deepshark
import subprocess as sp
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import getopt
import importlib as il
from glob import glob
from natsort import natsorted
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import MDS, Isomap
from MulticoreTSNE import MulticoreTSNE as TSNE
start=time.time()
#dimension1_2=16

#with gzip.open('/media/koh/HD-PCFU3/mouse/filter1_999_Tue_Oct_25_122720_2016.cpickle.gz', 'r') as f:
 #   saved_variables=cPickle.load(f)
 #   W_conv1, W_conv2, W_conv3, b_conv1, b_conv2, b_conv3, W_fc1, W_fc2, W_fc3, W_fc4, b_fc1, b_fc2, b_fc3, b_fc4=saved_variables



def roc_space_calc(label,pred):
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc

def softmax(w, t = 1.0):
    npa = np.array(w, dtype=np.float128)
    e = np.exp(npa / t,dtype=np.float128)
    dist = e /np.stack((np.sum(e, axis=1,dtype=np.float128),np.sum(e, axis=1,dtype=np.float128)),axis=-1)
    return dist
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

def batch_queuing(file_list, batch_size, data_length):
    half_batch=batch_size/2
    with tf.device('/cpu:0'):
        
        def process(f):
            with np.load(f) as f1:
                try:
                    dnase_data_labels=f1['labels'], f1['data_array']
                    
                except EOFError:
                    print("annot load: "+str(f))
            
            shape=dnase_data_labels[1].shape
            images=np.reshape(dnase_data_labels[1], (shape[0], data_length, 4, 1))
            labels=dnase_data_labels[0]            
            if shape>half_batch:
                halfimages=images[:half_batch] , images[half_batch:]
                halflabels=labels[:half_batch], labels[half_batch:]
            else:
                halfimages=images
                halflabels=labels
            
            return halfimages, halflabels
            
            
        image_list=[]
        label_list=[]
        #CPU=20
        #pool=mltp.Pool(CPU)
        for f in file_list:
            #res=apply_async(pool, process,args=(f,))
            #halfimages, halflabels=res.get()
            
            halfimages, halflabels=process(f)
            image_list.append(halfimages)
            label_list.append(halflabels)
        #pool.close()
        #pool.join()
        return image_list, label_list
 
def main():
    try:
        options, args =getopt.getopt(sys.argv[1:], 'i:o:n:b:t:g:c:G:', ['input_dir=','output_dir=','network_constructor=','bed=', 'test_genome=','genome_bed=','chromosome=','GPU='])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    if len(options)<4:
        print('too few argument')
        sys.exit(0)
    for opt, arg in options:
        if opt in ('-i', '--input_dir'):
            input_dir=arg
            if not os.path.isfile(input_dir):
                print(input_dir+' does not exist')
                sys.exit(0)
        elif opt in ('-o', '--output_dir'):
            output_dir=arg
        elif opt in ('-n', '--network_constructor'):
            model_name=arg
            #if not os.path.isfile(model_name):
                #print(model_name+' does not exist')

        elif opt in ('-t', '--test_genome'):
            test_genome=arg
            #if not os.path.isfile(test_genome):
                #print(test_genome+' does not exist')
                #sys.exit(0)
     
    input_dir_=input_dir.rsplit('.', 1)[0]
    path_sep=os.sep
    file_name=input_dir_.split(path_sep)
    a=time.asctime()
    b=a.replace(':', '')
    start_at=b.replace(' ', '_')
    out_dir=output_dir+file_name[-1]
    
    if not os.path.exists(os.path.dirname(out_dir)):
            try:
                os.makedirs(os.path.dirname(out_dir))
            except OSError as exc: # Guard against race condition
                if exc.errno != exc.errno.EEXIST:
                    raise
      
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)        

    x_image = tf.placeholder(tf.float32, shape=[None, 1000, 4, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, 20])
    keep_prob = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    keep_prob3 = tf.placeholder(tf.float32)
    phase=tf.placeholder(tf.bool)
    data_length=1000
    max_to_keep=2
    if 'ckpt' in input_dir.rsplit('.', 1)[1]: 
        input_dir=input_dir
    elif 'meta'  in input_dir.rsplit('.', 1)[1] or 'index'  in input_dir.rsplit('.', 1)[1]:
        input_dir=input_dir.rsplit('.', 1)[0]
    else:
        print("the input file should be a ckpt file")
        sys.exit(1)
    nc=il.import_module("deepgmap.network_constructors."+str(model_name))
    print("runing "+str(model_name))
    model = nc.Model(image=x_image, label=y_, 
                 output_dir=None,
                 phase=phase, 
                 start_at=None, 
                 keep_prob=keep_prob, 
                 keep_prob2=keep_prob2, 
                 keep_prob3=keep_prob3, 
                 data_length=data_length,
                 max_to_keep=max_to_keep)
    sess.run(tf.global_variables_initializer())
    saver=model.saver
    try:
        saver.restore(sess, input_dir)
    except:
        print("can't open "+str(input_dir))
        sys.exit(0)
    test_genome_list=natsorted(glob(test_genome))
    l=0
    position_list=[]
    y_prediction2=[]
    neuron_monitor_pos=[]
    BREAK=False
    for test_genome_ in test_genome_list:
        if BREAK:
            break
        genome_data=np.load(test_genome_)
        position_list_, seq_list=genome_data['positions'], genome_data['sequences']
        if len(position_list)==0:
            position_list=position_list_
        else:
            position_list=np.concatenate([position_list,position_list_])
        seq_list=np.array(seq_list, np.int16)
        seq_list=seq_list[np.random.randint(seq_list.shape[0], size=10000), :]
        seq_list=seq_list.reshape(-1, 1000, 4, 1)
        seq_length=seq_list.shape[0]
        print(seq_length)
        loop=seq_length/1000+1
        
        for i in range(loop):
            scanning=seq_list[i*1000:(i+1)*1000]
            if len(y_prediction2)==0:
                y_prediction2, active_neuron=sess.run([model.prediction[1],model.prediction[3]], feed_dict={x_image: scanning, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
                #neuron_monitor=active_neuron["h_pool3_flat"]/np.amax(active_neuron["h_pool3_flat"],axis=1).reshape([-1,1])
                neuron_monitor=active_neuron["h_pool3_flat"]
                print(y_prediction2.shape)
                print(neuron_monitor.shape)
                
            else:
                y_prediction1, active_neuron=sess.run([model.prediction[1],model.prediction[3]], feed_dict={x_image: scanning, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
                y_prediction2=np.concatenate([y_prediction2,y_prediction1],axis=0)
                #neuron_monitor=np.concatenate([neuron_monitor,active_neuron["h_pool3_flat"]/np.amax(active_neuron["h_pool3_flat"],axis=1).reshape([-1,1])],axis=0)
                neuron_monitor=np.concatenate([neuron_monitor,active_neuron["h_pool3_flat"]],axis=0)
            if (i+l*loop)%10==0:
                print(i+l*loop)
            if neuron_monitor.shape[0]>=10000:
                BREAK=True
                break
        l+=1
    sess.close()            
    
    print(neuron_monitor[0], np.max(neuron_monitor[0]))
    #print y_prediction2.shape    
    

    n_components = 2
    C=[]
    for i in range(len(y_prediction2)):
        C.append([np.mean(y_prediction2[i]),0.0,0.0])
    #print X_ipca.shape
    
    #pca = PCA(n_components=2)
    #X_pca = pca.fit_transform(neuron_monitor)
    #ism=Isomap()
    #X_pca = ism.fit_transform(neuron_monitor)
    #pca2 = PCA(n_components=50)
    #X_pca2 = pca2.fit_transform(neuron_monitor)
    tsne = TSNE(n_jobs=18,perplexity = 20.000000)
    neuron_monitor=np.array(neuron_monitor, np.float64)
    #X_pca2=np.array(X_pca2, np.float64)
    X_tsne = tsne.fit_transform(neuron_monitor)
    
    np.savez_compressed(str(out_dir)+"_np_arrays", neuron_monitor=neuron_monitor, transformed2=X_tsne)
    
    colors = ['navy', 'turquoise', 'darkorange']


    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
             lw=2, c=C,s=0.5)
    #err = np.abs(np.abs(X_pca) - np.abs(X_tsne)).mean()
    plt.title("TSNE of neuron activity")


    """
    for X_transformed, title in [(X_tsne, "TSNE"), (X_pca, "PCA")]:
        plt.figure(figsize=(16, 16))
        
    
        if "TSNE" in title:
            plt.scatter(X_transformed[:, 0], X_transformed[:, 1],
                     lw=2, c=C,s=0.5)
            #err = np.abs(np.abs(X_pca) - np.abs(X_tsne)).mean()
            plt.title(title + " of neuron activity")
        else:
            plt.scatter(X_transformed[:, 0], X_transformed[:, 1],
                     lw=2, c=C,s=0.5)
            plt.title(title + " of neuron activity")
        plt.legend(loc="best", shadow=False, scatterpoints=1)"""
        
    
    plt.show()

if __name__== '__main__':
    main()



