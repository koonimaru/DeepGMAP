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
        #print pred["prediction"].shape
        predictions_list.append(pred["prediction"])
        
    fpr_list=[]
    tpr_list=[]
    roc_auc_list=[]
    precision_list=[]
    recall_list=[]
    average_precision_list=[]
    label_array_shape= label_array.shape
    for i in predictions_list:
        for j in range(label_array_shape[1]):
            a, b=label_array[:,j], i[:,j]
            fpr, tpr, roc_auc=roc_space_calc(a, b)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(roc_auc)
            recall, precision,_ =precision_recall_curve(a, b)
            precision_list.append(precision)
            recall_list.append(recall)
            average_precision = average_precision_score(a, b)
            average_precision_list.append(average_precision)
            b+=0.15*np.random.randn(label_array_shape[0])
            b=np.clip(b, 0, 1)
            fpr, tpr, roc_auc=roc_space_calc(a, b)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(roc_auc)
            recall, precision,_ =precision_recall_curve(a, b)
            precision_list.append(precision)
            recall_list.append(recall)
            average_precision = average_precision_score(a, b)
            average_precision_list.append(average_precision)
       
    colormap = plt.cm.get_cmap('gnuplot')
    C = [colormap(i) for i in np.linspace(0,0.9,label_array_shape[1]*2)]
            
    plt.figure(1, figsize=(5,10))
    ax1=plt.subplot(211)
    
    
    #C=['darkorange','green','blue']
    i=0
    for fpr, tpr, roc_auc in zip(fpr_list,tpr_list,roc_auc_list):
        plt.plot(fpr, tpr, color=C[i],
              label=' (area = %0.2f)' % roc_auc)
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
    for prec, rec, avr_pr in zip(precision_list,recall_list,average_precision_list):
    
        plt.plot(prec, rec, lw=2, color=C[i],label=' (area = %0.2f)' % avr_pr)
        i+=1
    plt.axis('equal')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    ax2.set_ylim([0.0, 1.00])
    ax2.set_xlim([0.0, 1.0])
    
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower left")
    
    #plt.savefig(outfile_name, format='pdf')
    
    plt.show()
    
def main():
    outfile_name="/home/fast/onimaru/data/prediction/ROC_space_curve_comp_limb_brain.pdf"
    npload_list=[]
    label_array=[]
    chromosome="chr2"
    #name_list=["DeepSEA", "Bidirectional","Conv_plus","Conv+Bidirectional"]
    name_list=["conv4-FRSS"]
    file_list=["/home/fast2/onimaru/DeepGMAP-dev/data/predictions/conv4frss_Fri_Jun__8_101931_2018.ckpt-16747_prediction.npz"]
    label_file='/home/fast2/onimaru/DeepGMAP-dev/data/inputs/mm10_dnase_subset/dnase_subset_mm10_window1000_stride500.bed.labeled'
    
    with open(label_file, 'r') as fin:
        for line in fin:
            if line.startswith(chromosome):
                
                label_array.append(map(int, line.split()[3:]))
    label_array=np.array(label_array)
    
    for f in file_list:
        npload_list.append(np.load(f))
        
    
    roc_space_plotter(label_array, npload_list, name_list,outfile_name)


if __name__== '__main__':
    main()
    
    
    
    
    
    
    
    
    