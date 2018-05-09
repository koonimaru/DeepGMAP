from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
import time
import math
import os
#from natsort import natsorted, ns
#import subprocess as sp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
#from sklearn.preprocessing import label_binarize
#from sklearn.multiclass import OneVsRestClassifier
#from scipy import interp
import getopt
import importlib as il
from glob import glob
from natsort import natsorted
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
#from matplotlib import cm

start=time.time()

def roc_space_calc(label,pred):
    
    # Compute ROC curve and ROC area for each class
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
    
def run(args):
    main(args)

def main(args=None):
    TEST=True
    path_sep=os.path.sep
    #chromosome_of_interest='chr2'
    output_dir=None
    model_name=""
    bed_file=None
    max_to_keep=2
    GPU=1
      
    input_dir="/home/fast/onimaru/data/output/deepshark_Mon_Apr_23_115135_2018.ckpt-43995.meta"
    output_dir="/home/fast/onimaru/data/prediction/"
    model_name="deepshark"
    #bed_file=args.labeled_bed_file
    #test_genome=args.test_genome_files
    GPU=1
    #chromosome_of_interest=args.chromosome
    TEST=True

            
    if not os.path.isfile(input_dir):
        sys.exit(input_dir+' does not exist')
    #if not os.path.isfile(bed_file):
        #sys.exit(bed_file+' does not exist')

    if output_dir==None:
        sys.exit("output directory should be specified.")
    elif not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as exc:
            sys.exit(exc)
                
    input_dir_=input_dir.rsplit('.', 1)[0]
    sample_list=[]
    with open("/home/fast/onimaru/encode/deepsea/deepsea_pred.txt", 'r') as fin:
        for line in fin:
            line=line.split()
            sample_list.append(line[3])
 
    yshape=len(sample_list)
    data_length=1000
    
    if len(sample_list)==0:
        sample_list.append(os.path.basename(os.path.splitext(bed_file)[0]))
    
    test_array=np.load('/home/fast/onimaru/encode/deepsea/deepsea_train/test.npz')
    data_array, label_list=test_array["data_array"], test_array["labels"]
    
    path_sep=os.sep
    file_name=input_dir_.split(path_sep)
    a=time.asctime()
    b=a.replace(':', '')
    start_at=b.replace(' ', '_')
    out_dir=output_dir+file_name[-1]
    
    config = tf.ConfigProto(device_count = {'GPU': GPU})
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
        model_name=model_name[0]
    print("runing "+str(model_name))
    try:
        nc=il.import_module("deepgmap.network_constructors."+str(model_name))
    except ImportError:
        print(str(model_name)+" does not exist")
        sys.exit(0)
    """
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
    
    
    position_list=[]
    y_prediction2=[]
    
    #for test_genome_ in test_genome_list:
        #print(test_genome_)
        #genome_data=np.load(test_genome_)
        #position_list_, seq_list=genome_data['positions'], genome_data['sequences']

    seq_list=np.array(data_array, np.int16).reshape(-1, data_length, 4, 1)
    seq_length=seq_list.shape[0]
    print(seq_length)
    
    
    loop=int(math.ceil(float(seq_length)/1000))
    
    for i in range(loop):
        if i*1000>seq_length:
            break
        scanning=seq_list[i*1000:(i+1)*1000]
        if len(y_prediction2)==0:
            y_prediction2=sess.run(model.prediction[1], feed_dict={x_image: scanning, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
            
        else:
            y_prediction1=sess.run(model.prediction[1], feed_dict={x_image: scanning, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
            print(np.shape(y_prediction2), np.shape(y_prediction1))
            y_prediction2=np.concatenate([y_prediction2,y_prediction1],axis=0)
                                         
        if (i)%10==0:
            print(str(round(float(i)/loop*100, 0)), end="\r")

    sess.close()
    
    #saving the predictions as numpy array
    np.savez_compressed(str(out_dir)+"_prediction", prediction=y_prediction2)"""
    
    p=np.load("/home/fast/onimaru/data/prediction/deepshark_Mon_Apr_23_115135_2018.ckpt-43995_prediction.npz")
    y_prediction2=p["prediction"]
    #writing the predictions in narrowPeak format
    """
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
        output_handle.append(open(filename_1, 'w'))
    for i in range(len(y_prediction2)):
        a=position_list[i].strip('>')
        a=a.split(':')
        chrom=a[0]
        b=a[1].split('-')
        start_=b[0]
        end_=b[1]
        value=y_prediction2[i]
        for k in range(len(value)):
            output_handle[k].write("\t".join([str(chrom),str(start_),str(end_),'.',str(value[k]*1000).strip('[]'),'.',
                                              str(value[k]).strip('[]'),"-1\t-1\t-1\n"]))
            
    for i in output_handle:
        i.close()
    print('finished writing the prodictions in narrowPeak format to '+out_dir_np)"""
    
    if TEST==True:
        label_array=np.array(label_list, np.int16)
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
        
        index_=range(len(sample_list))
        
        sample_list, index_=zip(*sorted(zip(sample_list, index_)))
        
        fpr_list[:] = [fpr_list[i] for i in index_]
        tpr_list[:] = [tpr_list[i] for i in index_]
        roc_auc_list[:] = [roc_auc_list[i] for i in index_]
        recall_list[:] = [recall_list[i] for i in index_]
        precision_list[:] = [precision_list[i] for i in index_]
        average_precision_list[:] = [average_precision_list[i] for i in index_]
    
        mean_roc_auc=round(np.mean(roc_auc_list), 4)
        std_roc_auc=round(np.std(roc_auc_list), 4)
        max_roc_auc=round(np.amax(roc_auc_list), 4)
        min_roc_auc=round(np.amin(roc_auc_list), 4)
        mean_pre_rec=round(np.mean(average_precision_list), 4)
        std_pre_rec=round(np.std(average_precision_list), 4)
        max_pre_rec=round(np.amax(average_precision_list), 4)
        min_pre_rec=round(np.amin(average_precision_list), 4)       
        
        plt.figure(1, figsize=(14,14))
        ax1=plt.subplot(211)
        cmap = plt.get_cmap('gnuplot')
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
        # Shrink current axis by 20%
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
        plt.title('Receiver operating characteristic curve ('+str(model_name)+')')
        if len(sample_list)<20:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
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
        box2 = ax2.get_position()
        ax2.set_position([box2.x0, box2.y0, box2.width * 0.6, box2.height])
        plt.title('Precision-Recall curve ('+str(model_name)+')')
        if len(sample_list)<20:
        
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.savefig(out_dir+"ROC_space_curve.pdf", format='pdf')
        print(time.time()-start)
        
        with open(out_dir+"_prediction.log", 'w') as fo:
            fo.write("total time: "+str(time.time()-start)+"\ncommand: "+str(" ".join(sys.argv))+"\n"+
                     "roc auc (mean+-std): "+str(mean_roc_auc)+"+-"+str(std_roc_auc)+" with max "+str(max_roc_auc)+ " and min "+str(min_roc_auc) +"\n"+
                     "precision auc (mean+-std): "+ str(mean_pre_rec)+"+-"+str(std_pre_rec)+" with max "+str(max_pre_rec)+ " and min "+str(min_pre_rec)+"\n"+
                     "sample\troc_auc\tprecision_auc\n")
            for s, r, p in zip(sample_list,roc_auc_list, average_precision_list):
                fo.write(str(s)+"\t"+str(r)+"\t"+str(p)+"\n")
        
        plt.show()

if __name__== '__main__':
    main()
