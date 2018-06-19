import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

fname="/home/fast2/onimaru/DeepGMAP-dev/data/misc/mm10_ctcf_auprcs_frips.txt"

auprc=[]
frip=[]
"""
with open(fname, 'r') as fin:
    for line in fin:
        line=line.split()
        if len(line)>0:
            if line[0]=="AUPRC":
                auprc=map(float, line[1:])
            elif line[0]=="correctedFRiP":
                frip=map(float, line[1:])"""

with open(fname, 'r') as fin:
    for line in fin:
        line=line.split()
        if not len(line)==0 and not line[0]=="ID":
            auprc.append(float(line[1]))
            frip.append(float(line[5]))
frip, auprc=zip(*sorted(zip(frip, auprc)))
auprc_av=[]
for i in range(len(auprc)):
    auprc_av.append(np.average(auprc[i:]))

plt.figure(1, figsize=(4,4))
ax1=plt.subplot()
ax1.plot(frip,auprc_av)
ax1.grid(b=True, which='major', color='black', linestyle='-')
plt.xticks(np.arange(0, max(frip), 0.02))

ax1.grid(b=True, which='minor', color='gray', linestyle='--')
plt.minorticks_on()
plt.show()