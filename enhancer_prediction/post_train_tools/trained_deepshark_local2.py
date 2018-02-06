from __future__ import print_function
import sys
import gzip
import cPickle
import tensorflow as tf
import numpy as np
import time
import math
import os
from natsort import natsorted, ns
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
from matplotlib import cm

start=time.time()




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
    

def main():
    try:
        options, args =getopt.getopt(sys.argv[1:], 'i:o:n:b:t:g:c:', ['input_dir=','output_dir=','network_constructor=','bed=', 'test_genome=','genome_bed=','chromosome='])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    if len(options)<4:
        print('too few argument')
        sys.exit(0)
    path_sep=os.path.sep
    chromosome_of_interest='chr2'
    output_dir='./'
    model_name=""
    bed_file=None
    max_to_keep=2
    for opt, arg in options:
        if opt in ('-i', '--input_dir'):
            input_dir=arg
            if not os.path.isfile(input_dir):
                print(input_dir+' does not exist')
                sys.exit(0)
        elif opt in ('-o', '--output_dir'):
            output_dir=arg
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                except OSError as exc:
                    if exc.errno != exc.errno.EEXIST:
                        raise

        elif opt in ('-n', '--network_constructor'):
            model_name=arg
            #if not os.path.isfile(model_name):
                #print(model_name+' does not exist')
                #sys.exit(0)
        elif opt in ('-b', '--bed'):
            bed_file=arg
            if not os.path.isfile(bed_file):
                print(bed_file+' does not exist')
                sys.exit(0)
        elif opt in ('-t', '--test_genome'):
            test_genome=arg
            #if not os.path.isfile(test_genome):
                #print(test_genome+' does not exist')
                #sys.exit(0)
        elif opt in ('-g','--genome_bed'):
            genome_bed=arg
            if not os.path.isfile(genome_bed):
                print(genome_bed+' does not exist')
                sys.exit(0)
        elif opt in ('-c','--chromosome'):
            chromosome_of_interest=arg             
    input_dir_=input_dir.rsplit('.', 1)[0]
    sample_list=[]
    with open(bed_file, 'r') as fin:
        line=fin.readline()
        if "#" in line:
            line=line.split(": ")
            
            for i in line[1].split():
                s=i.split(path_sep)[-1].split(".")[0]
                sample_list.append(s)
            print(sample_list)
            line=fin.readline()
            
        line=line.split()
        
        yshape=len(line)-3
        if yshape==0:
            labeled_file=False
            yshape+=1
        else:
            labeled_file=True
        data_length=int(line[2])-int(line[1])
    
    if len(sample_list)==0:
        sample_list.append(bed_file.split(path_sep)[-1].split(".")[0])
    
    print(yshape)    
    
    if not labeled_file:
        #all_chromosome=False
        
        if chromosome_of_interest=="all":
            positive_region=set()
            with open(bed_file, 'r') as fpos:
                for line in fpos:
                    if not "#" in line:
                        positive_region.add(line)
            label_list=[]
            with open(genome_bed, "r") as fin:
                for line in fin:
    
                    if line in positive_region:
                        label_list.append(1)
                    else:
                        label_list.append(0)    
        else:
            positive_region=set()
            with open(bed_file, 'r') as fpos:
                for line in fpos:
                    if line.startswith(str(chromosome_of_interest)+"\t"):
                        positive_region.add(line)
            label_list=[]
            with open(genome_bed, "r") as fin:
                for line in fin:
                    if line.startswith(str(chromosome_of_interest)+"\t"):
                        if line in positive_region:
                            label_list.append(1)
                        else:
                            label_list.append(0)
    else:
        label_list=[]
        with open(bed_file, "r") as fin:
            if not chromosome_of_interest=="all":
                for line in fin:
                    if line.startswith(str(chromosome_of_interest)+"\t"):
                        line=line.split()
                        label_list.append(line[3:])
            else:
                for line in fin:
                    line=line.split()
                    label_list.append(line[3:])
                
    

    path_sep=os.sep
    file_name=input_dir_.split(path_sep)
    a=time.asctime()
    b=a.replace(':', '')
    start_at=b.replace(' ', '_')
    out_dir=output_dir+file_name[-1]
  
      
    config = tf.ConfigProto(device_count = {'GPU': 1})
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)       

    x_image = tf.placeholder(tf.float32, shape=[None, data_length, 4, 1])
    y_ = tf.placeholder(tf.float32, shape=[None, yshape])
    keep_prob = tf.placeholder(tf.float32)
    keep_prob2 = tf.placeholder(tf.float32)
    keep_prob3 = tf.placeholder(tf.float32)
    phase=tf.placeholder(tf.bool)
    
    if 'ckpt' in input_dir.rsplit('.', 1)[1]: 
        input_dir=input_dir
    elif 'meta'  in input_dir.rsplit('.', 1)[1] or 'index'  in input_dir.rsplit('.', 1)[1]:
        input_dir=input_dir.rsplit('.', 1)[0]
    else:
        print("the input file should be a ckpt file")
        sys.exit(1)
    if model_name=="":
        model_name=input_dir.split(path_sep)[-1].split("_")
        model_name="_".join(model_name[:4])
    print("runing "+str(model_name))
    try:
        nc=il.import_module("network_constructors."+str(model_name))
    except ImportError:
        print(str(model_name)+" does not exist")
        sys.exit(0)
    
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
    print(test_genome_list)
    l=0
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
        
        
        loop=int(math.ceil(float(seq_length)/1000))
        
        for i in range(loop):
            if i*1000>seq_length:
                break
            #print (i+1)*1000
            scanning=seq_list[i*1000:(i+1)*1000]
            if len(y_prediction2)==0:
                y_prediction2, active_neuron=sess.run([model.prediction[1],model.prediction[3]], feed_dict={x_image: scanning, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
                #neuron_monitor=active_neuron["h_fc1_drop"]
                #print y_prediction2.shape
                #print neuron_monitor.shape
                
            else:
                y_prediction1, active_neuron=sess.run([model.prediction[1],model.prediction[3]], feed_dict={x_image: scanning, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
                y_prediction2=np.concatenate([y_prediction2,y_prediction1],axis=0)
                #neuron_monitor=np.concatenate([neuron_monitor,active_neuron["h_fc1_drop"]],axis=0)
            if (i)%10==0:
                if l+1==1:
                    th='st'
                elif l+1==2:
                    th='nd'
                elif l+1==3:
                    th='rd'
                else:
                    th='th'
                print(str(float(i)/loop*100)+"% of "+str(l+1)+str(th)+" file has been scanned.", end="\r")
        l+=1
    sess.close()            

                             



    print(len(label_list))
    label_array=np.array(label_list, np.int16)
    
    
    #saving the predictions as numpy array
    np.savez_compressed(str(out_dir)+"_label_prediction", label_array=label_array, prediction=y_prediction2)
    
    #writing the predictions in narrowPeak format
    
    out_dir_np=out_dir+"_narrowPeaks/"
    
    if not os.path.exists(out_dir_np):
            try:
                os.makedirs(out_dir_np)
            except OSError as exc:
                if exc.errno != exc.errno.EEXIST:
                    raise
    output_handle=[]
    for s in sample_list:
        
        filename_1=out_dir_np+str(s)+'.narrowPeak'
        #print('writing '+filename_1)
        output_handle.append(open(filename_1, 'w'))
    
    #k=0
    for i in range(len(y_prediction2)):

        a=position_list[i].strip('>')
        #print(str(a)+'\t'+str(y_prediction2[i]))
        #k+=1
        a=a.split(':')
        chrom=a[0]
        b=a[1].split('-')
        start_=b[0]
        end_=b[1]
        value=y_prediction2[i]
        for k in range(len(value)):
            output_handle[k].write(str(chrom)+'\t'
                        +str(start_)+'\t'
                        +str(end_)+'\t.\t'
                        +str(value[k]*1000).strip('[]')+'\t.\t'
                        +str(value[k]).strip('[]')+"\t-1\t-1\t-1\n")
            
    #print("prediction num: "+str(k))
    for i in output_handle:
        i.close()
    print('finished writing '+filename_1)

    """
    for i in range(yshape) :
        if yshape==1:
            pred= y_prediction2[:]
            label=csr_matrix(label_array[:])
        else:
            pred= y_prediction2[:,i]
            label=label_array[:,i]
        pred_ = np.where(pred > 0.500000, np.ones_like(pred), np.zeros_like(pred))
        pred_1=csr_matrix(pred_)
        label1=csr_matrix(label)
        label2=csr_matrix(1*np.logical_not(label))
        pred_2=csr_matrix(1*np.logical_not(pred_))
        #print pred_
        tp = pred_1.dot(label1)
        fp = pred_1.dot(label2)
        fn = pred_2.dot(label1)
        tn = pred_2.dot(label2)
        PPV = np.true_divide(np.nansum(tp),
                         np.nansum(np.nansum(tp),np.nansum(fp)))
        TPR = np.true_divide(np.nansum(tp),
                         np.nansum(np.nansum(tp),np.nansum(fn)))

        print("F1 of "+str(i+1)+"th label = "+str(2*PPV*TPR/(PPV+TPR)))"""


    fpr_list=[]
    tpr_list=[]
    roc_auc_list=[]
    precision_list=[]
    recall_list=[]
    average_precision_list=[]
    if yshape>1:
        for i in range(yshape):
            fpr, tpr, roc_auc=roc_space_calc(label_array[:,i], y_prediction2[:,i])
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(roc_auc)
            precision, recall, _ = precision_recall_curve(label_array[:,i], y_prediction2[:,i])
            precision_list.append(precision)
            recall_list.append(recall)
            average_precision = average_precision_score(label_array[:,i], y_prediction2[:,i])
            average_precision_list.append(average_precision)
    else:
        fpr, tpr, roc_auc=roc_space_calc(label_array, y_prediction2)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)
        precision, recall, _ = precision_recall_curve(label_array, y_prediction2)
        precision_list.append(precision)
        recall_list.append(recall)
        average_precision = average_precision_score(label_array, y_prediction2)
        average_precision_list.append(average_precision)
        
    sample_list, fpr_list, tpr_list, roc_auc_list, recall_list,precision_list,average_precision_list=\
    zip(*sorted(zip(sample_list, fpr_list, tpr_list, roc_auc_list, recall_list,precision_list,average_precision_list)))
    
    mean_roc_auc=np.mean(roc_auc_list)
    mean_pre_rec=np.mean(average_precision_list)
    
    with open(out_dir+"_metrics.txt", 'w') as fo:
        fo.write("Mean roc auc: "+str(mean_roc_auc) +"\n"+
                 "Mean precision auc: "+ str(mean_pre_rec)+"\n"+
                 "Sample list: "+"\t".join(sample_list)+"\n"+
                 "roc auc list: "+"\t".join(map(str, roc_auc_list))+"\n"+
                 "precision auc list: "+"\t".join(map(str, average_precision_list)))
    
    
    
    plt.figure(1, figsize=(8,16))
    ax1=plt.subplot(211)
    cmap = plt.get_cmap('hot')
    colors = [cmap(i) for i in np.linspace(0, 1, len(sample_list))]
    
    i=0
    for i in range(yshape):
        f,t,r=fpr_list[i],tpr_list[i],roc_auc_list[i]
        plt.plot(f, t,color=colors[i],alpha=0.5,
            label=str(sample_list[i])+' (area = %0.2f)' % r)
        i+=1
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.axis('equal')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.title('Receiver operating characteristic curve ('+str(model_name)+')')
    plt.legend(loc="lower right")
    
    ax2=plt.subplot(212)
    i=0
    for i in range(yshape):
        r,p,a =recall_list[i],precision_list[i], average_precision_list[i]
        #print "F1: "+np.nanmean(2.0*r*p/(r+p))
        plt.plot(r, p, lw=2,alpha=0.5, color=colors[i],label=str(sample_list[i])+' (area = %0.2f)' % a)
        i+=1
    plt.axis('equal')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.00])
    plt.xlim([0.0, 1.0])
    
    plt.title('Precision-Recall curve ('+str(model_name)+')')
    plt.legend(loc="lower left")
    
    plt.savefig(out_dir+"ROC_space_curve.pdf", format='pdf')



    print(time.time()-start)
    plt.show()


if __name__== '__main__':
    main()



