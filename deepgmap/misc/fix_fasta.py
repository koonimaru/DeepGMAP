

import os
infile="/home/fast2/onimaru/1000genome/all_HG00119_H1.fa"
outfile="/home/fast2/onimaru/1000genome/all_HG00119_H1_edited.fa"


with open(infile, "r") as fin, open(os.path.split(infile)[0]+"/_tmp.fa", "w") as fo:
    i=0
    for line in fin:
        if line.startswith(">") and len(line.split())>1:
            line=line.split()[0]
            if i ==0:
                line=">chr"+line.strip(">")+"\n"
                i+=1
            else:
                line="\n"+">chr"+line.strip(">")+"\n"
            #print line
            fo.write(line)
        else:
            fo.write(line.strip("\n"))

#import numpy as np
dna=set(["A","G","C","T","N", "\n"])
with open(os.path.split(infile)[0]+"/_tmp.fa", "r") as fin, open(outfile, "w") as fo:
    
    
    for line in fin:
        if line.startswith(">"):
            fo.write(line)
            #print line
        else:
            i=0
            lline=(line)
            line=iter(line)
            seq=[]
            #i+=len(line)
            while True:
                try:
                    l=line.next()
                except StopIteration:
                    break
                #print l
                if l=="<":
                    l2=line.next()
                    while l2!=">":
                        l2=line.next()
                    
                    seq.append("N")
                    i+=1
                else:
                    seq.append(l)
                    i+=1
                if i%200==0:
                    seq.append("\n")
            fo.write("".join(seq))
                #if not any(l==dna):
                    

