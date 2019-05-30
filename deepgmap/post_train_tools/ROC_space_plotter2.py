import numpy as np

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from scipy import stats
import getopt
from glob import glob
from natsort import natsorted
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import matplotlib as mpl
import os
from decimal import Decimal
import pandas as pd

def roc_space_calc(label,pred):
    
    # Compute ROC curve and ROC area for each class

    fpr, tpr, _ = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc



def roc_space_plotter(label, predictions1,outfile_name):
    ind_list=['es', 'brain', 'limb']
    pos=[0,1]
    width=0.25
    predictions_list=[]
    label_array=np.array(label)
    label_array_shape=label_array.shape
    for pred in predictions1:
        print(pred["prediction"].shape)
        predictions_list.append(pred["prediction"])
    """df_rearanged = pd.DataFrame({
    ind_list[0] : [[], []],
    ind_list[1] : [[], []],
    ind_list[2] : [[], []],
    },index = ["deepsea", "conv4-frss"])"""
    
    data_dict={}
    for n, i in enumerate(predictions_list):
        if n<3:
            _key='deepsea'
            
        else:
            _key='conv4-frss'
            
        if not _key in data_dict:
                data_dict[_key]={}
        
        for j in range(label_array_shape[1]):
            
            _tmp_pred=np.where(i[:,j]>=0.5, 1,0)
            _tmp_label=label_array[:,j]
            #true_pos=((_tmp_pred+_tmp_label) ==2).sum()
            false_pos=((_tmp_label-_tmp_pred) <0).sum()
            #false_neg=((_tmp_label-_tmp_pred) ==1).sum()
            if not ind_list[j] in data_dict[_key]:
                data_dict[_key][ind_list[j]]=[]
            data_dict[_key][ind_list[j]].append(float(false_pos))
            
    for k in ind_list:
        a=data_dict['deepsea'][k]
        b=data_dict['conv4-frss'][k]
        s,p=stats.ttest_ind(a,b)
        print(p, k)
    
    
    
    df=pd.DataFrame(columns=["class1", "class2", "mean","stdv"])
    
    """class1=[]
    class2=[]
    name_of_class=["model","cell-type"]
    data_dict2={}"""
    for k, v in data_dict.items():
        #print k, v
        for k1,v1 in v.items():
            """for e in v1:
                class1.append(k)
                class2.append(k1)
                if not k in data_dict2:
                    data_dict2[k]=[]
                data_dict2[k].append(e)"""
    
            df=df.append({"class1":k, "class2":k1, "mean":np.mean(v1),"stdv":np.std(v1)}, ignore_index=True)
            
            
    """print data_dict2
    print class1
    print class2
    ix3 = pd.MultiIndex.from_arrays([class1, class2], names=name_of_class)
    df3 = pd.DataFrame(data_dict2, index=ix3)
    gp3 = df3.groupby(level=name_of_class)
    means = gp3.mean()
    errors = gp3.std()
    fig, ax = plt.subplots()
    means.plot.bar(yerr=errors, ax=ax)
    """
    #print df
    yerr=df.pivot(index='class2',columns='class1',values='stdv')
    #print np.shape(yerr)
    #print df.pivot(index='class2',columns='class1',values='mean')
    df.pivot(index='class2',columns='class1',values='mean').plot(kind='bar', yerr=yerr)
    
    #df.pivot(index='class1',columns='class2',values='mean').plot(kind='bar', yerr=df.std.reshape((2,3)))
    #print df.pivot(index='class1',columns='class2',values='std').values
    #df.pivot(index='class1',columns='class2',values='mean').plot(kind='bar')
    plt.grid(b=True, which='major', color='gray', linestyle='-',axis= 'y')
    plt.grid(b=True, which='minor', color='gray', linestyle='--',axis= 'y')
    plt.minorticks_on()
    #plt.grid(True)
    
    #plt.minorticks_on()
    plt.show()
            #print false_pos
            #print round(false_pos/np.float(false_pos+true_pos), 4)
            #print f1_score(label_array[:,j], )

    
def main():
    outfile_name="/home/fast/onimaru/data/prediction/ROC_space_curve_comp_limb_brain.pdf"
    npload_list1=[]
    npload_list2=[]
    label_array=[]
    label_array_append=label_array.append
    chromosome="chr2"
    #name_list=["DeepSEA", "Bidirectional","Conv_plus","Conv+Bidirectional"]
    name_list=["conv4frss", "deepsea"]
    file_list1=[
                "/home/fast2/onimaru/DeepGMAP-dev/data/predictions/quick_benchmark/deepsea_Fri_Apr_20_140717_2018.ckpt-16747_prediction.npz",
                "/home/fast2/onimaru/DeepGMAP-dev/data/predictions/deepsea_Thu_Jun__7_072332_2018.ckpt-16747_prediction.npz",
                "/home/fast2/onimaru/DeepGMAP-dev/data/predictions/quick_benchmark/deepsea_Thu_Apr_26_115030_2018.ckpt-16747_prediction.npz",
                '/home/fast2/onimaru/DeepGMAP-dev/data/predictions/conv4frss_Fri_Jun__8_101931_2018.ckpt-16747_prediction.npz',
                '/home/fast2/onimaru/DeepGMAP-dev/data/predictions/conv4frss_Fri_Jun__8_122816_2018.ckpt-16747_prediction.npz',
                "/home/fast2/onimaru/DeepGMAP-dev/data/predictions/quick_benchmark/deepsharktest_Thu_Apr_19_191806_2018.ckpt-16747_prediction.npz",
              ]
    label_file_array="/home/fast/onimaru/deepgmap/data/inputs/mm10_dnase_subset/dnase_summits_subset_mm10_1000_chr2_testlabels.npz"
    if not os.path.isfile(label_file_array):
        label_file='/home/fast/onimaru/deepgmap/data/inputs/mm10_dnase_subset/dnase_summits_subset_mm10_1000.bed.labeled'
        with open(label_file, 'r') as fin:
            for line in fin:
                if line.startswith(chromosome):
                    line=line.split()
                    #print line
                    label_array_append(map(int, line[3:]))
        label_array=np.array(label_array)
        np.savez_compressed( label_file_array,  labels=label_array,)
    else:
        label_array=np.load(label_file_array)["labels"]
    for f in file_list1:
        npload_list1.append(np.load(f))
    
    roc_space_plotter(label_array, npload_list1,outfile_name)


if __name__== '__main__':
    main()
    
    
    
    
    
    
    
    
    