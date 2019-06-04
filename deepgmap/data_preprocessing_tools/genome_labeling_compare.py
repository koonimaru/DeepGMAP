
import glob as glb
import sys
import numpy as np
from sklearn.decomposition import KernelPCA as pca_f
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch
from MulticoreTSNE import MulticoreTSNE as TSNE

def genome_label(bed_file_list, genome_1000):
    file_num=len(bed_file_list)
    peak_set_list=[]
    i=0
    for f in bed_file_list:
        peak_set=set()
        with open(f, 'r') as fin:
            for line in fin:
                if i==0:
                    _,a,b=line.split()
                    check_length=int(b)-int(a)
                    
                peak_set.add(line)
        peak_set_list.append(peak_set)
                
        i+=1
    label_array_list=[]
    with open(genome_1000,'r') as fin:
        i=0
        for line in fin:
            k=0
            label_array=[0 for h in range(file_num)]

            for s in peak_set_list:
                if i==0:
                    _,a,b=line.split()
                    assert check_length==int(b)-int(a), "mismatches in sequence lengths"
                if line in s:
                    label_array[k]=1
                k+=1 
            if sum(label_array)>0:
                #print sum(label_array)
                label_array_list.append(label_array)
            i+=1
    return np.array(label_array_list)

def main():
    bed_file_dir, genome_1000=sys.argv[1:]
    bed_file_list=[]
    if not "*" in bed_file_dir and bed_file_dir.endswith('.bed'):
        bed_file_list.append(bed_file_dir)
    elif not '*' in bed_file_dir:
        bed_file_dir=bed_file_dir+"*.bed"
    
    bed_file_list=glb.glob(bed_file_dir)
    #print bed_file_list
    if len(bed_file_list)==0:
       # print("no files in "+str(bed_file_dir))
        sys.exit()
    label_array_list=genome_label(bed_file_list, genome_1000)
    #print label_array_list[0]
    label_array_list=label_array_list[np.random.randint(label_array_list.shape[0], size=5000), :]
    
    
    label_array_list_=np.transpose(label_array_list)
    #print sum(label_array_list_[0])
    lshape=label_array_list.shape
    C=[]
    for i in range(lshape[0]):
        
        C.append([np.sum(label_array_list[i])/float(lshape[1]),0.0,0.0])
    tsne = TSNE(n_jobs=18,perplexity = 5.000000)
    label_array_list=np.array(label_array_list, np.float64)
    #X_pca2=np.array(X_pca2, np.float64)
    X_tsne = tsne.fit_transform(label_array_list)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
             c=C, lw=2, s=0.5)
    
    pca = pca_f(n_components=2, kernel="rbf")
    X_pca=pca.fit_transform(label_array_list)
    dist1=pdist(label_array_list_, 'cosine')
    _, ax1=plt.subplots()
    
    Y = sch.linkage(dist1, method='ward')
    Z1 = sch.dendrogram(Y)
    idx1 = Z1['leaves']
    
    new_sample_list=[]
    
    for i in idx1:
        txt=bed_file_list[i].split("/")[-1]
        new_sample_list.append(txt)
    ax1.set_xticklabels(new_sample_list , rotation=90)
    
    
    #print X_pca.shape
    _, ax2=plt.subplots()
    ax2.scatter(X_pca[:,0], X_pca[:,1],c=C)
    """for i, txt in enumerate(bed_file_list):
        txt=txt.split("/")[-1]
        ax2.annotate(txt, (X_pca[i,0],X_pca[i,1]))"""
    
    plt.show()
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
