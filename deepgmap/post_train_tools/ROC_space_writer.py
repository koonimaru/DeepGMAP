import sys
import gzip
import cPickle
import tensorflow as tf
import numpy as np
import time
import math
import os
from natsort import natsorted, ns
import network_constructors.network_constructor_deepsea_1d3 as nc
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

def roc_space_calc(label,pred):
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc


file_list=['/home/fast/onimaru/data/prediction/\
network_constructor_deepsea_1d3_Wed_Aug_16_072343_2017_step7923.ckpt-7923_label_prediction.npz',
'/home/fast/onimaru/data/prediction/\
network_constructor_deepsea_1d_Wed_Aug_16_101507_2017_step6348.ckpt-6348_label_prediction.npz',
'/home/fast/onimaru/data/prediction/\
network_constructor_danq_1d_Wed_Aug_16_105449_2017_step7815.ckpt-7815_label_prediction.npz',
'/home/fast/onimaru/human/fimo_out_1e3/fimo_prediction.npz']

pred_list=[]

i=0
for f in file_list:
    npld=np.load(f)
    if i==0:
        label_array=npld["label_array"]
    pred_list.append(npld["prediction"])
    i+=1

fpr_list=[]
tpr_list=[]
roc_auc=[]
precision_list=[]
recall_list=[]

for pred in pred_list:
    fpr1, tpr1, roc_auc1=roc_space_calc(label_array, pred)
    precision1, recall1, _ = precision_recall_curve(label_array, pred)
    fpr_list.append(fpr1)
    tpr_list.append(tpr1)
    roc_auc.append(roc_auc1)
    precision_list.append(precision1)
    recall_list.append(recall1)

plt.figure(1, figsize=(8,16))
ax1=plt.subplot(211)
plt.plot(fpr1, tpr1, color='darkorange',
          label='DeepShark (area = %0.2f)' % roc_auc1)
plt.plot(fpr2, tpr2, color='green',
          label='DeepSEA (area = %0.2f)' % roc_auc2)
plt.plot(fpr3, tpr3, color='blue',
          label='Danq (area = %0.2f)' % roc_auc3)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.axis('equal')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")

ax2=plt.subplot(212)
plt.plot(recall1, precision1, lw=2, color='darkorange',label='DeepShark (area = %0.2f)' % average_precision1)
plt.plot(recall2, precision2, lw=2, color='green',label='DeepSEA (area = %0.2f)' % average_precision2)
plt.plot(recall3, precision3, lw=2, color='blue',label='Danq (area = %0.2f)' % average_precision3)
plt.axis('equal')
plt.xlabel('Recall')
plt.ylabel('Precision')
ax2.set_ylim([0.0, 1.00])
ax2.set_xlim([0.0, 1.0])

plt.title('Precision-Recall curve')
plt.legend(loc="lower left")

plt.savefig("/home/fast/onimaru/data/prediction/ROC_space_curve_comp_mESC_CTCF.pdf", format='pdf')

plt.show()