import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
def pr_curve_writer(label, pred):
    
    a=len(label)
    
    b=380000
    curve_resolution=10000
    linspace=np.linspace(0.0000, 1.0, curve_resolution,endpoint=True, dtype=np.float64)
    TPR_array=np.zeros([curve_resolution], dtype=np.float64)
    FPR_array=np.zeros([curve_resolution], dtype=np.float64)
    PPV_array=np.zeros([curve_resolution], dtype=np.float64)
    if a>=b:      
        label1=csr_matrix(label)
        label2=csr_matrix(1*np.logical_not(label))
        
        print('calculating the first ROC space')

        for i in range(curve_resolution):
            print("creating binary array")
            pred_ = np.where(pred >= linspace[i], np.ones_like(pred), np.zeros_like(pred))
            pred2=1*np.logical_not(pred_)
            #pred_=csc_matrix(pred_)
            #print pred_
            #print "calc logical and"
            tp = label1.dot(pred_)
            
            #print sum(tp)
            
            fp = label2.dot(pred_)
            #print fp
            fn = label1.dot(pred2)
            #print fn
            tn = label2.dot(pred2)
            #print tn
            
            FPR_array[i] += np.true_divide(fp,tn+fp)
            TPR_array[i] += np.true_divide(tp,tp+fn)
            if tp+fp==0.0:
                PPV_array[i]+=0.0
            else:
                PPV_array[i] += np.true_divide(tp,tp+fp)
            #print i
            #if i>=curve_resolution-3:
                #print TPR_array[i],PPV_array[i]
        
    else:
        for i in range(curve_resolution):
            pred_ = np.where(pred >= linspace[i], np.ones_like(pred), np.zeros_like(pred))
            #print pred_
            tp = np.logical_and(pred_, label)
            fp = np.logical_and(pred_, np.logical_not(label))
            fn = np.logical_and(np.logical_not(pred_), label)
            tn = np.logical_and(np.logical_not(pred_), np.logical_not(label))
            FPR_array[i] = np.true_divide(np.nansum(fp),
                             np.nansum(np.logical_or(tn, fp)))
            TPR_array[i] = np.true_divide(np.nansum(tp),
                             np.nansum(np.logical_or(tp, fn)))
            if np.nansum(np.logical_or(tp, fp))==0.0:
                PPV_array[i]=0.0
            else:
                PPV_array[i] = np.true_divide(np.nansum(tp),
                             np.nansum(np.logical_or(tp, fp)))
                
            #if i>=curve_resolution-3:
                #print TPR_array[i],PPV_array[i]
            #rint i
    area=0.0
    k=curve_resolution-1
    for i in range(curve_resolution):
        area+=0.500*(PPV_array[k]+PPV_array[k-1])*(TPR_array[k-1]-TPR_array[k])
        #print area
        k-=1
        if k==0:
            break
    
            
    return FPR_array, TPR_array, PPV_array, area

array_file='/home/fast/onimaru/data/prediction/network_constructor_deepsea_1d3_Tue_Sep_19_150851_2017.ckpt-10734_label_prediction.npz'
#genome_bed=''
np_in=np.load(array_file)
pred=np_in["prediction"]
#print len(pred)
label_array=np_in["label_array"]
#print pred[:,0]
if len(label_array.shape)==1:
    num_label=1
else:
    num_label=label_array.shape[1]

fpr_list=[]
tpr_list=[]
roc_auc_list=[]
precision_list=[]
recall_list=[]
average_precision_list=[]
if num_label>1:
    for i in range(num_label):
        
    
        fpr, tpr, ppv, area=pr_curve_writer(label_array[:,i], pred[:,i])
        precision_list.append(ppv)
        #tpr_list.append(tpr)
        recall_list.append(tpr)
        average_precision_list.append(area)
else:
    fpr, tpr, ppv, area=pr_curve_writer(label_array, pred)

    precision_list.append(ppv)
    recall_list.append(tpr)
    average_precision = area
    average_precision_list.append(average_precision)
plt.figure(1, figsize=(8,8))
"""ax1=plt.subplot(211)
i=0
for i in range(num_label):
    f,t,r=fpr_list[i],tpr_list[i],roc_auc_list[i]
    plt.plot(f, t, color='darkorange',
        label='ROC curve ('+str(i)+') (area = %0.2f)' % r)
    i+=1
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.axis('equal')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic curve ('+str(model_name)+')')
plt.legend(loc="lower right")"""

#ax2=plt.subplot(212)
i=0
for i in range(num_label):
    r,p,a =recall_list[i],precision_list[i], average_precision_list[i]
    plt.plot(r, p, lw=2, color='navy',label='Precision-Recall curve ('+str(i)+') (area = %0.2f)' % a)
    i+=1
plt.axis('equal')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.00])
plt.xlim([0.0, 1.0])

#plt.title('Precision-Recall curve ('+str(model_name)+')')
plt.legend(loc="lower left")

#plt.savefig(out_dir+"ROC_space_curve_"+str(model_name)+".pdf", format='pdf')


plt.show()
