import scipy.stats as sc

f="/home/fast2/onimaru/DeepGMAP-dev/data/misc/cfrip_mm10_ctcf.txt"

peaks=[]
frips=[]
cfrips=[]

with open(f, "r") as fin:
    for line in fin:
        line=line.split()
        if not line[0]=="ID":
            peaks.append(float(line[4]))
            frips.append(float(line[2]))
            cfrips.append(float(line[5]))

#print sc.spearmanr(frips, peaks)
#print sc.spearmanr(cfrips, peaks)