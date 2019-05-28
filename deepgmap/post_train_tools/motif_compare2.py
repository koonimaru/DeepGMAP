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
    h,t=os.path.split(motif_data_dir)
    foutname=h+"/"+os.path.splitext(t)[0]+"tmp.meme"
    with open(foutname, "w") as fo, open(motif_data_dir, 'r') as fin:
        
        fo.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\nBackground letter\
frequencies (from uniform background):\nA 0.2500 C 0.2500 G 0.2500 T 0.2500\n\n")
        lines=fin.readlines()
        for i, line in enumerate(lines):
            if line.startswith("letter-probability"):
                start_line=i+1
                break
        for i in range(100-2):
            fo.write("MOTIF tmp_"+str(i*10)+"-"+str(i*10+30)+"\n\nletter-probability matrix: alength= 4 w= 30 nsites= 30 E= 0\n")
            for l in lines[start_line+i*10:start_line+i*10+30]:
                fo.write(l)
            fo.write("\n\n")
            
    return foutname
                

def main():
    motif_data_dir="/home/fast/onimaru/data/meme/merged.meme"
    #long_motif_dir="/home/fast/onimaru/deepgmap/data/reconstructions/conv4frss_Fri_May_11_075425_2018.ckpt-16747Tue_May_15_112518_2018_all_.pdf.meme"
    long_motif_dir="/home/fast2/onimaru/DeepGMAP-dev/data/activation_max/conv4frss_Fri_Sep_28_160038_2018.ckpt-28907Thu_Dec_20_131413_2018_ese14_re.pdf.meme"
    #fout=os.path.splitext(long_motif_dir)[0]+".matches"
    #fout="/home/fast/onimaru/data/output/network_constructor_deepsea_1d3_Fri_Oct_13_133809_2017.ckpt-15899Mon_Oct_16_105338_2017.npz.matches"

    fname=motif_reader(long_motif_dir)
    #print fname
if __name__== '__main__':
    main()                
                