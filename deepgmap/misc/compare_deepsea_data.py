import numpy as np
import matplotlib.pyplot as plt

deepsea="/home/fast/onimaru/encode/deepsea/deepsea_pred.txt"

deepshark="/home/fast/onimaru/encode/deepsea/deepshark_Tue_Apr_17_183529_2018.ckpt-57883_prediction.log"

deepsea_dict={}

with open(deepsea, 'r') as fin:
    for line in fin:
        if not line.startswith("Cell Type"):
            #print line
            line=line.split()
            if len(line)==0:
                continue
            print(line)
            if line[4]=="NA":
                continue
            sname=line[3].split('.')[0]
            AUPRC=float(line[5])
            deepsea_dict[sname]=AUPRC

sample_list=[]
deepsea_list=[]
deepshark_list=[]  
with open(deepshark, 'r') as fin:
    go=False
    for line in fin:
        if line.startswith("sample"):
            go=True
            continue
        elif go:
            line=line.split()
            sname=line[0].split("_")[0]
            if "Dnase" in sname and sname in deepsea_dict:
                sample_list.append(sname)
                deepsea_list.append(deepsea_dict[sname])
                deepshark_list.append(float(line[2]))
                print(sname, deepsea_dict[sname], float(line[2]))

deepsea_list=np.array(deepsea_list)
deepshark_list=np.array(deepshark_list)

log_fold=np.log2(deepshark_list/deepsea_list)
log_fold_neg=log_fold[log_fold<0.00]
print("total num: "+str(len(log_fold))+"\nless performed num:"+str(len(log_fold_neg))+" ("+str(len(log_fold_neg)/float(len(log_fold))*100.0)+"%)")
