
import glob as glb
import sys
import numpy as np
from sklearn.decomposition import KernelPCA as pca_f
import os
import matplotlib as mpl
mpl.use("WebAgg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch
import time
import copy

def genome_label(bed_file_list, genome_1000,out_dir):
    
    file_num=len(bed_file_list)

    #print file_num
    peak_set_list=[]
    peak_set_list_append=peak_set_list.append
    #start=time.time()
    i=0
    for f in bed_file_list:
        peak_set=set()
        peak_set_add=peak_set.add
        with open(f, 'r') as fin:
        
        
            for line in fin:
                if i==0:
                    _,a,b=line.split()
                    check_length=int(b)-int(a)
                    
                peak_set_add(line)
            peak_set_list_append(peak_set)
                    
        i+=1
    
    fo_name=out_dir
    label_array_list=[]
    label_array_list_append=label_array_list.append
    with open(genome_1000,'r') as fin:
        with open(fo_name,'w') as fout:
            fout.write("#sample_list: "+"\t".join(bed_file_list)+"\n")
            i=0
            
            for line in fin:
                k=0
                label_array=["0" for h in range(file_num)]
    
                for s in peak_set_list:
                    if i==0:
                        _,a,b=line.split()
                        assert check_length==int(b)-int(a), "mismatches in sequence lengths"
                    if line in s:
                        label_array[k]="1"
                    k+=1 
                fout.write(line.strip('\n')+'\t'+' '.join(label_array)+'\n')
                #if sum(label_array)>0:
                    #abel_array_list_append(label_array)
                i+=1
                if i%200000==0:
                    
                    sys.stdout.write("\rwriting labeled file "+ line.strip("\n"))
                    sys.stdout.flush()
    #print time.time()-start
    #sys.exit()
    print("\n"+fo_name+" has been saved. This file is going to be used when testing a trained model too.")
    #return label_array_list


def genome_label2(bed_file_list, genome_1000,out_dir):
    
    file_num=len(bed_file_list)

    #print file_num
    peak_set_dict={}
    #peak_set_list_append=peak_set_list.append
    start=time.time()
    i=0
    #zero=["0" for h in range(file_num)]
    for f in bed_file_list:
        with open(f, 'r') as fin:
            for line in fin:
                if i==0:
                    _,a,b=line.split()
                    check_length=int(b)-int(a)

                if not line in peak_set_dict:
                    peak_set_dict[line]=["0" for h in range(file_num)]
                    #peak_set_dict[line]=copy.deepcopy(zero)
                peak_set_dict[line][i]="1"
        i+=1
    print(time.time()-start)
    fo_name=out_dir
    label_array_list=[]
    label_array_list_append=label_array_list.append
    zero=' '.join(["0" for h in range(file_num)])
    with open(genome_1000,'r') as fin:
        with open(fo_name,'w') as fout:
            fout.write("#sample_list: "+"\t".join(bed_file_list)+"\n")
            #start=time.time()
            i=0
            for line in fin:
                if i==0:
                    _,a,b=line.split()
                    assert check_length==int(b)-int(a), "mismatches in sequence lengths"
                if line in peak_set_dict:
                    fout.write(line.strip('\n')+'\t'+' '.join(peak_set_dict[line])+'\n')
                else:
                    fout.write(line.strip('\n')+'\t'+zero+'\n')
                #if sum(label_array)>0:
                    #label_array_list_append(label_array)
                i+=1
                if i%200000==0:
                    
                    sys.stdout.write("\rwriting labeled file "+ line.strip("\n"))
                    sys.stdout.flush()
    print("genome_labeling2 "+str(time.time()-start))
    #sys.exit()
    print("\n"+fo_name+" has been saved. This file is going to be used when testing a trained model too.")
    #return label_array_list


def main():
    #bed_file_dir, genome_1000, out_dir=sys.argv[1:]
    bed_file_dir="/home/fast/onimaru/deepgmap/data/inputs/hg38_dnase/peaks_10k/test_hg38_window1000_stride300.bed_list/*"
    genome_1000="/home/fast/onimaru/deepgmap/data/genomes/hg38_window1000_stride300.bed"
    out_dir="/home/fast/onimaru/deepgmap/data/inputs/hg38_dnase/peaks_10k/test.labeled"
    bed_file_list=[]
    if not "*" in bed_file_dir and bed_file_dir.endswith('.bed'):
        bed_file_list.append(bed_file_dir)
    elif not '*' in bed_file_dir:
        bed_file_dir=bed_file_dir+"*.bed"
    
    bed_file_list=sorted(glb.glob(bed_file_dir))
    print(bed_file_list)
    if len(bed_file_list)==0:
        print("no files in "+str(bed_file_dir))
        sys.exit()
    label_array_list=genome_label2(bed_file_list, genome_1000,out_dir)
    print(label_array_list[0])
    label_array_list_=np.transpose(label_array_list)
    print(sum(label_array_list_[0]))
    pca = pca_f(n_components=2, kernel="rbf")
    X_pca=pca.fit_transform(label_array_list_)
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
    
    
    print(X_pca.shape)
    _, ax2=plt.subplots()
    ax2.scatter(X_pca[:,0], X_pca[:,1])
    for i, txt in enumerate(bed_file_list):
        txt=txt.split("/")[-1]
        ax2.annotate(txt, (X_pca[i,0],X_pca[i,1]))
    
    #plt.show()
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
