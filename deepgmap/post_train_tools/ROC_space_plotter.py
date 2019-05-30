import numpy as np

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import getopt
from glob import glob
from natsort import natsorted
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib as mpl

def roc_space_calc(label,pred):
    
    # Compute ROC curve and ROC area for each class

    fpr, tpr, _ = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc



def roc_space_plotter(label, predictions, name_list,outfile_name):
    predictions_list=[]
    label_array=label
    for pred in predictions:
        print(pred["prediction"].shape)
        predictions_list.append(pred["prediction"])
        
    fpr_list=[]
    tpr_list=[]
    roc_auc_list=[]
    precision_list=[]
    recall_list=[]
    average_precision_list=[]
    for i in predictions_list:
        fpr, tpr, roc_auc=roc_space_calc(label_array, i)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        roc_auc_list.append(roc_auc)
        recall, precision,_ =precision_recall_curve(label_array, i)
        precision_list.append(precision)
        recall_list.append(recall)
        average_precision = average_precision_score(label_array, i)
        average_precision_list.append(average_precision)
        
       
       
    colormap = plt.cm.get_cmap('gnuplot')
    #C = [colormap(i) for i in np.linspace(0,0.9,len(name_list))]
            
    plt.figure(1, figsize=(5,10))
    ax1=plt.subplot(211)
    

    C=['darkorange','green','blue']
    i=0
    for fpr, tpr, roc_auc,name in zip(fpr_list,tpr_list,roc_auc_list,name_list):
        plt.plot(fpr, tpr, color=C[i],
              label=str(name)+' (area = %0.2f)' % roc_auc)
        i+=1
    plt.plot([0, 1], [0, 1.0], color='navy', linestyle='--')
    plt.axis('equal')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    
    ax2=plt.subplot(212)
    i=0
    for prec, rec, avr_pr,name in zip(precision_list,recall_list,average_precision_list,name_list):
    
        plt.plot(prec, rec, lw=2, color=C[i],label=str(name)+' (area = %0.2f)' % avr_pr)
        i+=1
    plt.axis('equal')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    ax2.set_ylim([0.0, 1.00])
    ax2.set_xlim([0.0, 1.0])
    
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    
    plt.savefig(outfile_name, format='pdf')
    
    plt.show()
    
def main():
    outfile_name="/home/fast/onimaru/data/prediction/ROC_space_curve_comp_limb_brain.pdf"
    npload_list=[]
    label_array=[]
    chromosome="chr2"
    #name_list=["DeepSEA", "Bidirectional","Conv_plus","Conv+Bidirectional"]
    name_list=["DeepSEA", "DanQ","Conv+Bidirectional"]
    file_list=['/home/fast/onimaru/data/prediction/network_constructor_danq_1d_Sat_Nov_18_151721_2017.ckpt-12123_label_prediction.npz',
               #'/home/fast/onimaru/data/prediction/network_constructor_deepsea_1d4_Fri_Oct__6_183716_2017.ckpt-11467_label_prediction.npz',
               "/home/fast/onimaru/data/prediction/network_constructor_danq_1d_Sat_Nov_18_151721_2017.ckpt-12123_label_prediction.npz",
               "/home/fast/onimaru/data/prediction/network_constructor_deepsea_1d3_Fri_Nov_17_170434_2017.ckpt-12123_label_prediction.npz"]
    label_file=''
    
    with open(label_file, 'r') as fin:
        for line in fin:
            if line.startswith(chromosome):
                label_array.append(map(int, line[3:]))
    label_array=np.array(label_array)
    
    for f in file_list:
        npload_list.append(np.load(f))
        
    
    roc_space_plotter(label_array, npload_list, name_list,outfile_name)


if __name__== '__main__':
    main()
    
    
    
    
    
    
    
    
    