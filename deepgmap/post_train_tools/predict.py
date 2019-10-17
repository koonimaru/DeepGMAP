from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
import time
import math
import os
#import subprocess as sp
import matplotlib as mpl
mpl.use("WebAgg")
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
PATH_SEP=os.path.sep
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
    chromosome_of_interest='all'
    output_dir=None
    model_name=""
    bed_file=None
    max_to_keep=2
    #GPU=1
    if args is None:
        sys.exit("please specify required options.")
        #takes arguments when the code is run through enhancer_prediction_run
    TEST=args.test_or_prediction
    GPUID=str(args.gpuid)
    BATCH_SIZE=args.batchsize
    test_genome=args.test_genome_files
    WRITE_PRED=args.write_prediction
    if args.logfile=="":
           
        input_dir=args.input_ckpt
        output_dir=args.out_directory
        model_name=args.model
        bed_file=args.labeled_bed_file
        _prefix=args.prefix
        if not args.chromosome =="None":
            chromosome_of_interest=args.chromosome
        if bed_file=="":
            if input_dir.endswith(".meta"):
                _logfile=os.path.splitext(os.path.splitext(input_dir)[0])[0]+".log"
            else:
                _logfile=os.path.splitext(input_dir)[0]+".log"
            with open(_logfile, "r") as fin:
                for line in fin:
    
                    if line.startswith("Labeled file:"):
                        bed_file=line.split(":")[1].strip(" \n")
        if bed_file=="":
            sys.exit("please specify -b option.")
    else:
        if not args.logfile.endswith(PATH_SEP):
            args.logfile+=PATH_SEP
        #WORKDIR=os.path.split(os.path.split(args.logfile)[0])[0]
        output_dir=args.logfile+"predictions"+PATH_SEP
        _prefix=args.prefix
        input_dir=natsorted(glob(args.logfile+"train*.meta"))[-1]
        print('saved models are '+str(input_dir))
        with open(args.logfile+"train.log", "r") as fin:
            for line in fin:
                """
                if line.startswith('The last check point:'):
                    input_dir=line.split(":")[1].strip(" \n")
                    if not os.path.isfile(input_dir):
                        input_dir=os.path.split(args.logfile)[0]+"/"+os.path.split(input_dir)[1]
                        if not os.path.isfile(input_dir):
                            sys.exit("unable to find checkpoint file ("+os.path.split(input_dir)[1]+")")
                            
                    #input_dir=line[1]
                """
                if line.startswith("Labeled file:"):
                    bed_file=line.split(":")[1].strip(" \n")
                        
                    #bed_file=line[1]
                elif line.startswith("Model:"):
                    model_name=line.split(":")[1].strip(" \n")
                elif line.startswith("Excluded"):
                    chromosome_of_interest=line.split(":")[1].strip(" \n").strip("'")
                    if "," in chromosome_of_interest:
                        c1, c2=chromosome_of_interest.split(', ')
                        chromosome_of_interest=c1.strip("'")+","+c2.strip("'")
        if args.chromosome is not "None":
            chromosome_of_interest=args.chromosome
        
        print(chromosome_of_interest)
                    #model_name=line[1]
                    
    
    if not os.path.isfile(input_dir):
        sys.exit('the input file named '+input_dir+' does not exist.')
    if chromosome_of_interest=="all":
        TEST=False
    if output_dir==None:
        sys.exit("output directory should be specified.")
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    
    
    
    
    
    if not test_genome.endswith(PATH_SEP):
        test_genome=test_genome+PATH_SEP+"*.npz"
    test_genome_list=natsorted(glob(test_genome))
    if len(test_genome_list)==0:
        sys.exit('test genome named '+test_genome+" does not exist.")
                
    input_dir_=input_dir.rsplit('.', 1)[0]
    sample_list=[]
    with open(bed_file, 'r') as fin:
        line=fin.readline()
        if "#" in line:
            line=line.split(": ")
            
            for i in line[1].split():
                #s=i.split(path_sep)[-1].split(".")[0]
                s=os.path.basename(os.path.splitext(i)[0])
                
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
        sample_list.append(os.path.basename(os.path.splitext(bed_file)[0]))
    
    print(yshape)
    if "," in chromosome_of_interest:
        chromosome_of_interest=set(chromosome_of_interest.split(','))
    else:
        chromosome_of_interest=set([chromosome_of_interest])
        
        
    
    """"if not labeled_file:
        if not "all" in chromosome_of_interest:
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
    else:"""
    print(chromosome_of_interest)
    if not "all" in chromosome_of_interest:
        label_list=[]
        with open(bed_file, "r") as fin:
            for line in fin:
                line=line.split()
                #print(line[0])
                if line[0] in chromosome_of_interest:
                    #print(line[0], chromosome_of_interest)
                    label_list.append(line[3:])

    path_sep=os.sep
    file_name=input_dir_.split(path_sep)
    a=time.asctime()
    b=a.replace(':', '')
    start_at=b.replace(' ', '_')
    out_dir=output_dir+_prefix+"-"+file_name[-1]
    
    config = tf.ConfigProto()
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
        model_name=input_dir.split(path_sep)[-2].split("_")
        model_name=model_name[0]
    print("runing "+str(model_name))
    
    nc=il.import_module("deepgmap.network_constructors."+str(model_name))

    
    model = nc.Model(image=x_image, label=y_, 
                 output_dir=None,
                 phase=phase, 
                 start_at=None, 
                 keep_prob=keep_prob, 
                 keep_prob2=keep_prob2, 
                 keep_prob3=keep_prob3, 
                 data_length=data_length,
                 max_to_keep=max_to_keep,
                 GPUID=GPUID)
    sess.run(tf.global_variables_initializer())
    saver=model.saver
    #try:
    saver.restore(sess, input_dir)
    #except:
        #print("can't open "+str(input_dir))

    
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
        
        
        loop=int(math.ceil(float(seq_length)/BATCH_SIZE))
        
        for i in range(loop):
            if i*BATCH_SIZE>seq_length:
                break
            scanning=seq_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            if len(y_prediction2)==0:
                y_prediction2, active_neuron=sess.run([model.prediction[1],model.prediction[3]], feed_dict={x_image: scanning, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
                
            else:
                y_prediction1, active_neuron=sess.run([model.prediction[1],model.prediction[3]], feed_dict={x_image: scanning, keep_prob: 1.0, keep_prob2: 1.0, keep_prob3: 1.0,phase:False})
                y_prediction2=np.concatenate([y_prediction2,y_prediction1],axis=0)
                                             
            if (i)%10==0:
                if l+1==1:
                    th='st'
                elif l+1==2:
                    th='nd'
                elif l+1==3:
                    th='rd'
                else:
                    th='th'
                print(str(round(float(i)/loop*100, 0))+"% of "+str(l+1)+str(th)+" file has been scanned.", end="\r")
        l+=1
    sess.close()
    
    #saving the predictions as numpy array
    np.savez_compressed(str(out_dir)+"_prediction", prediction=y_prediction2)
    
    #writing the predictions in narrowPeak format
    if WRITE_PRED:
        out_dir_np=out_dir+"_narrowPeaks"+PATH_SEP
        print('Writing the prodictions in narrowPeak format to '+out_dir_np)
        
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
            a=str(position_list[i]).strip('>')
            a=a.split(':')
            chrom=a[0]
            b=a[1].split('-')
            start_=b[0]
            end_=b[1]
            value=y_prediction2[i]
            for k in range(len(value)):
                output_handle[k].write("\t".join([str(chrom),str(start_),str(end_),
                                                  '.',str(value[k]*1000).strip('[]'),'.',
                                                  str(value[k]).strip('[]'),"-1\t-1\t-1\n"]))
                
        for i in output_handle:
            i.close()
        print('finished writing the predictions.')
    
    if TEST==True:
        print("Now, calculating AUROC and AUPRC scores...")
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
                #average_precision = auc(recall, precision)
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
        
        print("Drawing the curves...")
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
        print("done. predictions have been written in a directory, "+os.path.split(out_dir)[0]+".")
        #plt.show()

if __name__== '__main__':
    main()
