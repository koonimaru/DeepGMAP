import random
import os
labeledf="/home/fast/onimaru/data/Chip-seq/narrowPeaks/three_3times.labeled"

h, t =os.path.split(labeledf)

pos_num=0
neg_num=0

with open(labeledf, 'r') as fin:
    for line in fin:
        if line.startswith("#"):
            continue
        line=line.split()
        i=sum(map(int, line[3:]))
        if i >0:
            pos_num+=1
        else:
            neg_num+=1
            

#print pos_num, neg_num

r=float(pos_num)/(0.75*neg_num)

with open(labeledf, 'r') as fin, open(h+"/down_sampled_"+str(round(r,4))+"_"+t, "w") as fl, open(h+"/down_sampled_"+str(round(r,4))+"_"+t+".bed", "w") as fb:
    for line in fin:
        if line.startswith("#"):
            fl.write(line)
            continue
        line_=line.split()
        i=sum(map(int, line_[3:]))
        rand=random.random()
        
        if i > 0:
            fl.write(line)
            fb.write("\t".join(line_[:3])+"\n")
        elif rand<r:
            fl.write(line)
            fb.write("\t".join(line_[:3])+"\n")
            