
file_name="/home/fast/onimaru/data/CTCF/mm10_CTCF_qc2_mm10_1000.bed.labeled"
from random import shuffle

a=file_name.split('.')[0]+"_shuffled.bed.labeled"

with open(file_name, "r") as fin, open(a, "w") as fo:
    i=0
    for line in fin:
        if line.startswith("#"):
            line=line.split()
            list_of_label=line[1:]
            
            x = [i for i in range(len(list_of_label))]
            shuffle(x)
            list_of_label_shuf=[]
            for e in x:
                list_of_label_shuf.append(list_of_label[e])
            fo.write(line[0]+" "+" ".join(list_of_label_shuf)+"\n")
        else:
            b=line.split()
            pos="\t".join(b[:3])
            tmp=b[3:]
            tmp_shuf=[]
            #print tmp
            tmp2=map(int, tmp)
            if sum(tmp2)>0:
            
                for e in x:
                    tmp_shuf.append(tmp[e])
            else:
                tmp_shuf=tmp
            label=" ".join(tmp_shuf)
            fo.write(pos+"\t"+label+"\n")

