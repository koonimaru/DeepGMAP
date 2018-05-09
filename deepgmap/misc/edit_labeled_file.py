
lf="/home/fast/onimaru/encode/mm10_dnase-seq_subset/deepsea_type_wondow_mm10_s200.bed.labeled"
gf="/home/fast/onimaru/data/genome_fasta/mm10.genome"

chrm_dict={}

with open(gf, 'r') as fin:
    for line in fin:
        line=line.split()
        chrm_dict[line[0]]=int(line[1])
import os
h, t=os.path.split(lf)
elf=h+"/edited_"+t
with open(lf,'r') as fin, open(elf,'w') as fo:
    for line in fin:
        if line.startswith("#"):
            fo.write(line)
        else:
            line=line.split()
            start=int(line[1])-400
            end=int(line[2])+400
            if start>=0 and end<=chrm_dict[line[0]]:
                fo.write('\t'.join([line[0],str(start),str(end)])+"\t"+" ".join(line[3:])+"\n")
