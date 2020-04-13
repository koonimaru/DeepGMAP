import numpy as np
import sys
#from curses.ascii import isdigit
from scipy.spatial.distance import cdist
import deepgmap.post_train_tools.cython_util as cutil
mc=cutil.motif_compare
from matplotlib import pyplot as plt 
import os
def _is_number(s):
    try:
        complex(s) # for int, long, float and complex
    except ValueError:
        return False

    return True

def motif_reader(motif_data_dir):
    motif_name=""
    motif_dict={}
    motif_list=[]
    with open(motif_data_dir, 'r') as fin:
        MOTIF=False
        i=0
        for line in fin:
            i+=1
            line=line.split()
            if len(line)==0:
                MOTIF=False
                continue
            elif line[0]=="MOTIF":
                if len(motif_name)>0:
                    motif_dict[motif_name]=np.array(motif_list)
                    
                motif_list=[]
                motif_name=""
                if len(line)>2:
                    motif_name="_".join(line[1:])
                else:
                    motif_name=line[1]
                
            elif line[0]=="letter-probability":
                if line[4]=="w=":
                    motif_length=int(line[5])
                else:
                    print("something wrong in line "+str(i))
                    sys.exit()
                MOTIF=True
            elif MOTIF==True:
                #print _is_number(line[0])
                if not _is_number(line[0]):
                    MOTIF=False
                    continue
                else:
                    motif_list.append(map(float, line))
        
        motif_dict[motif_name]=np.array(motif_list)
    return motif_dict




def motif_compare(motif_data_dict, long_motif_dict, fout, THRESHOLD=-5.0):
    with open(fout, "w") as f:
        f.write("Motif name\tStart\tEnd\tdistance\n")
        for k1, v1 in long_motif_dict.items():
            
            v1shape=v1.shape
            #print v1
            j=0
            for k2, v2 in motif_data_dict.items():
                if "secondary" in k2:
                    continue
                #print k2
                #j+=1
                #print j
                v2shape=v2.shape
                RAND_DIST=[]
                for i in range(12):
                    rand=np.random.rand(v2shape[0],v2shape[1])
                    for k in range(v2shape[1]):
                        rand[k]=rand[k]/np.sum(rand[k])
                    RAND_DIST.append(np.mean(np.diagonal(cdist(v2, rand,metric='cosine'))))
                RAND_MEAN=np.mean(RAND_DIST)
                RAND_DEV=np.std(RAND_DIST)
                #print RAND_MEAN, RAND_DEV
                #print("randome_dist: "+str(RAND_DIST))
                
                
                
                for i in range(v1shape[0]-v2shape[0]):
                    partial_motif=v1[i:(i+v2shape[0])]
                    #print v2shape, partial_motif.shape
                    """M=0.5*(partial_motif+v2)+0.00001
                    JSD=0.5*(np.sum(-v2*np.log(M/(v2+0.00001)))+np.sum(-partial_motif*np.log(M/(partial_motif+0.00001))))/v2shape[0]
                    print JSD"""
                    DIST=np.mean(np.diagonal(cdist(v2, partial_motif,metric='cosine')))
                    Z_SCORE=(DIST-RAND_MEAN)/RAND_DEV
                    #print Z_SCORE
                    if Z_SCORE<=THRESHOLD:
                        f.write(str(k2)+"\t"+str(i)+"\t"+str(i+v2shape[0])+"\t"+str(Z_SCORE)+"\n")

def main():
    motif_data_dir="/home/fast/onimaru/data/meme/merged.meme"
    #long_motif_dir="/home/fast/onimaru/deepgmap/data/reconstructions/conv4frss_Fri_May_11_075425_2018.ckpt-16747Tue_May_15_112518_2018_all_.pdf.meme"
    long_motif_dir="/home/fast/onimaru/deepgmap/data/reconstructions/conv4frss_Fri_May_11_075425_2018.ckpt-16747Tue_May_15_104419_2018_es_e14_.pdf.meme"
    fout=os.path.splitext(long_motif_dir)[0]+".matches"
    #fout="/home/fast/onimaru/data/output/network_constructor_deepsea_1d3_Fri_Oct_13_133809_2017.ckpt-15899Mon_Oct_16_105338_2017.npz.matches"
    motif_data_dict=motif_reader(motif_data_dir)
    #print len(motif_data_dict)
    long_motif_dict=motif_reader(long_motif_dir)
    #print len(long_motif_dict)
    #motif_compare(motif_data_dict, long_motif_dict, fout)
    Z_SCORE_list=mc(motif_data_dict, long_motif_dict, fout, THRESHOLD=-5)
    plt.hist(Z_SCORE_list, 1000)
    plt.xticks(np.arange(min(Z_SCORE_list), max(Z_SCORE_list)+1, 1.0))
    plt.show()

if __name__== '__main__':
    main()                
                